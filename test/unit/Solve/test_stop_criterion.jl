using XCALibre
using Accessors
using LinearAlgebra
using SparseArrays
using Test

# Every serial solve path stops when ‖D⁻¹(b - Ax)‖₂ ≤ atol + rtol·‖D⁻¹(b - Ax₀)‖₂, D = |diag(A)|,
# whatever the solver, preconditioner or precision. The system is scaled symmetrically by factors
# spanning four orders of magnitude, so other norms (‖b - Ax‖₂, √(rᵀM⁻¹r), ‖Mr‖) differ widely.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "finer_mesh_laplace.unv"))
hardware = Hardware(backend=CPU(), workgroup=1024)

model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=mesh)
BCs = assign(region=mesh, (T = [
    Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall), Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
schemes = (T = Schemes(laplacian = Linear),)

function scaled_system(solver, preconditioner, precision; rtol, atol, itmax=1000)
    setup = SolverSetup(solver=solver, preconditioner=preconditioner, convergence=1e-8, relax=1.0,
        rtol=rtol, atol=atol, itmax=itmax, precision=precision)
    config = Configuration(solvers=(T = setup,), schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1), hardware=hardware, boundaries=BCs)
    T = model.energy.T
    T_eqn = (
        - Laplacian{schemes.T.laplacian}(model.solid.rDf, T) == - Source(ScalarField(mesh))
    ) → ScalarEquation(T, config.boundaries.T)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)
    A, b = XCALibre.ModelFramework._A(T_eqn), XCALibre.ModelFramework._b(T_eqn)
    d = 10 .^ (2 .* sin.(1:length(b)))
    rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
    for i ∈ eachindex(b), nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
        nzval[nzi] *= d[i]*d[colval[nzi]]
    end
    b .*= d
    @reset T_eqn.preconditioner = set_preconditioner(setup, T_eqn)
    @reset T_eqn.solver = _workspace(setup, T_eqn)
    update_preconditioner!(T_eqn.preconditioner, mesh, config)
    T_eqn, setup, T, config
end

function true_residual(T_eqn, x)
    A = parent(XCALibre.ModelFramework._A(T_eqn))
    norm((XCALibre.ModelFramework._b(T_eqn) - A*x) ./ abs.(diag(sparse(A))))
end

# warm start: the reference for rtol is b - Ax₀, not b
warm!(T) = T.values .= 20 .+ 10 .* cos.(3 .* (1:length(T.values)))

function solve_warm!((T_eqn, setup, T, config))
    r0 = true_residual(T_eqn, warm!(T))
    solve_system!(T_eqn, setup, T, nothing, config)
    r0, true_residual(T_eqn, T.values)
end

const CASES = (
    (Cg(), Jacobi(), FullPrecision()), (Cg(), DILU(), FullPrecision()),
    (Bicgstab(), Jacobi(), FullPrecision()), (Bicgstab(), DILU(), FullPrecision()),
    (Cgs(), Jacobi(), FullPrecision()), (Gmres(), Jacobi(), FullPrecision()), (Gmres(), DILU(), FullPrecision()),
    (Cg(), Jacobi(), MixedF32()), (Cg(), DILU(), MixedF32()), (Bicgstab(), DILU(), MixedF32()))
case_name((s, p, pr)) = "$(nameof(typeof(s))) $(nameof(typeof(p))) $(nameof(typeof(pr)))"

# Float32 corrections meet the target of their own Float32 residual; the Float64 one differs by rounding
slack(::FullPrecision) = 1.0
slack(_) = 1.05

@testset "$(case_name(c)) stops at rtol of the Jacobi-scaled residual" for c ∈ CASES
    rtol = 1e-3
    r0, r = solve_warm!(scaled_system(c...; rtol=rtol, atol=0.0))
    @test r ≤ slack(c[3])*rtol*r0
    # stops near the target, not on a preconditioned norm far below it
    @test r ≥ 1e-3*rtol*r0
end

# r0 is the same for every case: one system, one warm start
const R0 = let (T_eqn, _, T, _) = scaled_system(Cg(), Jacobi(), FullPrecision(); rtol=0.0, atol=0.0)
    true_residual(T_eqn, warm!(T))
end

@testset "$(case_name(c)) stops at atol of the Jacobi-scaled residual" for c ∈ CASES
    atol = 1e-4*R0
    r0, r = solve_warm!(scaled_system(c...; rtol=0.0, atol=atol))
    @test r0 ≈ R0
    @test r ≤ slack(c[3])*atol
end

# MixedF16 solves the system scaled to unit diagonal; atol still applies to D⁻¹(b - Ax). Its Float32
# corrections reach ~3e-3 of R0 here, where D⁻¹ weights rows over four orders of magnitude
@testset "MixedF16 Jacobi stops at atol and rtol of the Jacobi-scaled residual" begin
    _, r = solve_warm!(scaled_system(Cg(), Jacobi(), MixedF16(); rtol=1e-2, atol=0.0))
    @test r ≤ 1.05e-2*R0
    _, r = solve_warm!(scaled_system(Cg(), Jacobi(), MixedF16(); rtol=0.0, atol=1e-2*R0))
    @test r ≤ 1e-2*R0
end
