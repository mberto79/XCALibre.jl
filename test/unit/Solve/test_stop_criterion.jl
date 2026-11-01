using XCALibre
using Accessors
using LinearAlgebra
using SparseArrays
using Test

# Every serial solve path stops when ‖D⁻¹(b - Ax)‖₂ ≤ atol + rtol·‖D⁻¹(b - Ax₀)‖₂, D = |diag(A)|,
# whatever the solver, preconditioner or precision. The system is scaled symmetrically by factors
# spanning four orders of magnitude, so other norms (‖b - Ax‖₂, √(rᵀM⁻¹r), ‖Mr‖) differ widely.
# test/gpu/test_stop_criterion.jl runs the same checks on the device.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "finer_mesh_laplace.unv"))
schemes = (T = Schemes(laplacian = Linear),)

function scaled_system(solver, preconditioner, precision; rtol, atol, itmax=1000, backend=CPU(), scaled=true)
    dmesh = backend isa CPU ? mesh : adapt(backend, mesh)
    model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=dmesh)
    BCs = assign(region=dmesh, (T = [
        Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall), Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
    setup = SolverSetup(solver=solver, preconditioner=preconditioner, convergence=1e-8, relax=1.0,
        rtol=rtol, atol=atol, itmax=itmax, precision=precision)
    config = Configuration(solvers=(T = setup,), schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=backend, workgroup=backend isa CPU ? 1024 : 32), boundaries=BCs)
    T = model.energy.T
    T_eqn = (
        - Laplacian{schemes.T.laplacian}(model.solid.rDf, T) == - Source(ScalarField(dmesh))
    ) → ScalarEquation(T, config.boundaries.T)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)
    A, b = XCALibre.ModelFramework._A(T_eqn), XCALibre.ModelFramework._b(T_eqn)
    d = scaled ? 10 .^ (2 .* sin.(1:length(b))) : ones(length(b))
    rowptr, colval, nzval, bh = Array(_rowptr(A)), Array(_colval(A)), Array(_nzval(A)), Array(b)
    for i ∈ eachindex(bh), nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
        nzval[nzi] *= d[i]*d[colval[nzi]]
    end
    copyto!(_nzval(A), nzval)
    copyto!(b, bh .* d)
    @reset T_eqn.preconditioner = set_preconditioner(setup, T_eqn)
    @reset T_eqn.solver = _workspace(setup, T_eqn)
    update_preconditioner!(T_eqn.preconditioner, dmesh, config)
    T_eqn, setup, T, config
end

# ‖D⁻¹(b - Ax)‖₂ on the host
function true_residual(T_eqn, x)
    A = XCALibre.ModelFramework._A(T_eqn)
    rowptr, colval, nzval = Array(_rowptr(A)), Array(_colval(A)), Array(_nzval(A))
    b, x = Array(XCALibre.ModelFramework._b(T_eqn)), Array(x)
    s = 0.0
    for i ∈ eachindex(b)
        r, aii = b[i], 0.0
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            r -= nzval[nzi]*x[colval[nzi]]
            colval[nzi] == i && (aii = nzval[nzi])
        end
        s += (r/abs(aii))^2
    end
    sqrt(s)
end

# warm start: the reference for rtol is b - Ax₀, not b
warm!(T) = (copyto!(T.values, 20 .+ 10 .* cos.(3 .* (1:length(T.values)))); T.values)

function solve_warm!((T_eqn, setup, T, config))
    r0 = true_residual(T_eqn, warm!(T))
    solve_system!(T_eqn, setup, T, nothing, config)
    r0, true_residual(T_eqn, T.values)
end

# stationary AMG needs constant near-null vectors, which the scaling removes: it runs unscaled below
const AMG_MODES = (Cg(), Bicgstab())
const CASES = (
    (Cg(), Jacobi(), FullPrecision()), (Cg(), DILU(), FullPrecision()),
    (Bicgstab(), Jacobi(), FullPrecision()), (Bicgstab(), DILU(), FullPrecision()),
    (Cgs(), Jacobi(), FullPrecision()), (Cgs(), DILU(), FullPrecision()),
    (Gmres(), Jacobi(), FullPrecision()), (Gmres(), DILU(), FullPrecision()),
    (Cg(), Jacobi(), MixedF32()), (Cg(), DILU(), MixedF32()), (Bicgstab(), DILU(), MixedF32()),
    (Cgs(), DILU(), MixedF32()), (Gmres(), DILU(), MixedF32()),
    ((s, AMG(), FullPrecision()) for s ∈ AMG_MODES)...)
case_name((s, p, pr)) = p isa AMG ? "AMG $(nameof(typeof(s)))" :
    "$(nameof(typeof(s))) $(nameof(typeof(p))) $(nameof(typeof(pr)))"

# Float32 corrections meet the target of their own Float32 residual; the Float64 one differs by rounding
slack(::FullPrecision) = 1.0
slack(_) = 1.05

function check_stop_criterion(cases; backend=CPU())
    @testset "$(case_name(c)) stops at rtol of the Jacobi-scaled residual" for c ∈ cases
        rtol = 1e-3
        r0, r = solve_warm!(scaled_system(c...; rtol=rtol, atol=0.0, backend))
        @test r ≤ slack(c[3])*rtol*r0
        # stops near the target, not on a preconditioned norm far below it
        @test r ≥ 1e-3*rtol*r0
    end

    # r0 is the same for every case: one system, one warm start
    R0 = let (T_eqn, _, T, _) = scaled_system(Cg(), Jacobi(), FullPrecision(); rtol=0.0, atol=0.0, backend)
        true_residual(T_eqn, warm!(T))
    end

    @testset "$(case_name(c)) stops at atol of the Jacobi-scaled residual" for c ∈ cases
        atol = 1e-4*R0
        r0, r = solve_warm!(scaled_system(c...; rtol=0.0, atol=atol, backend))
        @test r0 ≈ R0
        @test r ≤ slack(c[3])*atol
    end

    # atol and rtol add: atol = ρ·r0 with rtol = ρ is the same target as rtol = 2ρ alone, so the
    # iterates match (a max of the two would stop at half the target). GPU AMG sums in no fixed order
    @testset "$(case_name(c)) adds atol and rtol" for c ∈ cases
        ρ = 1e-3
        _, r_sum = solve_warm!(scaled_system(c...; rtol=ρ, atol=ρ*R0, backend))
        _, r_ref = solve_warm!(scaled_system(c...; rtol=2ρ, atol=0.0, backend))
        (backend isa CPU || !(c[2] isa AMG)) && @test r_sum ≈ r_ref rtol=1e-6
        @test r_sum ≤ slack(c[3])*2ρ*R0
    end

    # x₀ already within atol: the field is returned untouched (Gmres and Cgs leave this to Krylov.jl)
    @testset "$(case_name(c)) keeps a converged x₀" for c ∈ cases
        T_eqn, setup, T, config = scaled_system(c...; rtol=0.0, atol=10*R0, backend)
        x0 = Array(warm!(T))
        solve_system!(T_eqn, setup, T, nothing, config)
        @test Array(T.values) == x0
    end

    # MixedF16 solves the system scaled to unit diagonal; atol still applies to D⁻¹(b - Ax). Its Float32
    # corrections reach ~3e-3 of R0 here, where D⁻¹ weights rows over four orders of magnitude
    @testset "MixedF16 Jacobi stops at atol and rtol of the Jacobi-scaled residual" begin
        _, r = solve_warm!(scaled_system(Cg(), Jacobi(), MixedF16(); rtol=1e-2, atol=0.0, backend))
        @test r ≤ 1.05e-2*R0
        _, r = solve_warm!(scaled_system(Cg(), Jacobi(), MixedF16(); rtol=0.0, atol=1e-2*R0, backend))
        @test r ≤ 1e-2*R0
    end
end

function check_amg_solver(; backend=CPU())
    @testset "AMG Richardson stops at the target (unscaled system)" begin
        sys(rtol, atol) = scaled_system(Richardson(), AMG(), FullPrecision(); rtol, atol, backend, scaled=false)
        r0, r = solve_warm!(sys(1e-3, 0.0))
        @test 1e-6*r0 ≤ r ≤ 1e-3*r0
        _, r = solve_warm!(sys(0.0, 1e-4*r0))
        @test r ≤ 1e-4*r0
        _, r_sum = solve_warm!(sys(1e-3, 1e-3*r0))
        _, r_ref = solve_warm!(sys(2e-3, 0.0))
        backend isa CPU && @test r_sum ≈ r_ref rtol=1e-6
        @test r_sum ≤ 2e-3*r0
    end
end

# test/gpu/test_stop_criterion.jl includes this file for its helpers and runs its own cases
if !@isdefined(STOP_CRITERION_GPU)
    check_stop_criterion(CASES)
    check_amg_solver()
end
