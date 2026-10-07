using XCALibre
using Accessors
using LinearAlgebra
const Krylov = XCALibre.Solve.Krylov

# Repeated MixedPrecision solves are iterative refinement: each corrects the full-precision residual,
# so the iterate must reach the Float64 solution although every Krylov solve runs in low precision.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "finer_mesh_laplace.unv"))

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)

model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=mesh)
BCs = assign(region=mesh, (T = [
    Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall), Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
schemes = (T = Schemes(laplacian = Linear),)

function mp_solve_T(precision, nsolves; rtol=0.1, preconditioner=Jacobi())
    solvers = (T = SolverSetup(
        solver=Cg(), preconditioner=preconditioner, convergence=1e-8, relax=1.0, rtol=rtol, itmax=1000,
        precision=precision),)
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1), hardware=hardware, boundaries=BCs)
    T = model.energy.T
    T_eqn = (
        - Laplacian{schemes.T.laplacian}(model.solid.rDf, T) == - Source(ScalarField(mesh))
    ) → ScalarEquation(T, config.boundaries.T)
    initialise!(T, 0.0)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)
    @reset T_eqn.preconditioner = set_preconditioner(solvers.T.preconditioner, T_eqn)
    @reset T_eqn.solver = _workspace(solvers.T, T_eqn)
    update_preconditioner!(T_eqn.preconditioner, mesh, config)
    for _ ∈ 1:nsolves
        solve_system!(T_eqn, solvers.T, T, nothing, config)
    end
    copy(T.values)
end

mp_reference = mp_solve_T(FullPrecision(), 1; rtol=1e-12)

@testset "MixedPrecision($TL) refines to the Float64 solution" for TL ∈ (BFloat16, Float16, Float32)
    # BFloat16's 8-bit mantissa contracts the error ~0.8x per solve (Float16 ~0.4x), hence many solves
    x = mp_solve_T(MixedPrecision(TL), 160)
    @test norm(x - mp_reference)/norm(mp_reference) < 1e-8
end

@testset "MixedPrecision setup checks" begin
    @test_throws ArgumentError mp_solve_T(MixedPrecision(), 1; preconditioner=DILU())
    @test MixedPrecision() isa MixedPrecision{Float32}
    @test SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0).precision isa FullPrecision
    @test_throws ArgumentError SolverSetup(
        solver=AMG(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0, precision=MixedPrecision())
end

@testset "half-precision XVector dot accumulates in Float32" begin
    n = 100_000
    x = XCALibre.Multithread.XVector(ones(BFloat16, n))
    @test Float64(Krylov.kdot(n, x, x)) ≈ n rtol=1e-2
end
