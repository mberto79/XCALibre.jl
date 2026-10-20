using XCALibre
using Accessors
using Base.CoreLogging: Warn, with_logger
using Test

# A linear solve that stops without meeting atol/rtol warns on every serial path (Krylov, mixed
# precision with one correction or with iterative refinement, AMG), with the reason, iteration
# count and ‖b - Ax‖₂; a solve that converges does not warn.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "finer_mesh_laplace.unv"))
hardware = Hardware(backend=CPU(), workgroup=1024)

model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=mesh)
BCs = assign(region=mesh, (T = [
    Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall), Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
schemes = (T = Schemes(laplacian = Linear),)

function laplace_system(solver, precision; itmax, rtol, atol, preconditioner=Jacobi())
    solvers = (T = SolverSetup(solver=solver, preconditioner=preconditioner, convergence=1e-8, relax=1.0,
        rtol=rtol, atol=atol, itmax=itmax, precision=precision),)
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1), hardware=hardware, boundaries=BCs)
    T = model.energy.T
    T_eqn = (
        - Laplacian{schemes.T.laplacian}(model.solid.rDf, T) == - Source(ScalarField(mesh))
    ) → ScalarEquation(T, config.boundaries.T)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)
    @reset T_eqn.preconditioner = set_preconditioner(solvers.T.preconditioner, T_eqn)
    @reset T_eqn.solver = _workspace(solvers.T, T_eqn)
    update_preconditioner!(T_eqn.preconditioner, mesh, config)
    T_eqn, solvers.T, T, config
end

function solve_from_zero!((T_eqn, setup, T, config))
    initialise!(T, 0.0)
    solve_system!(T_eqn, setup, T, nothing, config)
end

laplace_solve(solver, precision; kwargs...) = solve_from_zero!(laplace_system(solver, precision; kwargs...))

const AMG_PC = AMG(coarsening=SmoothAggregation(), smoother=AMGJacobi())

@testset "unconverged $(nameof(typeof(s))) $(nameof(typeof(p))) solve warns" for (s, p, rtol, msg, pc) ∈ (
        (Cg(), FullPrecision(), 0.0, r"Cg solve did not converge in 2 iterations \(maximum number of iterations exceeded\): ‖b - Ax‖₂ = ", Jacobi()),
        (Bicgstab(), FullPrecision(), 0.0, r"Bicgstab solve did not converge in 2 iterations \(maximum number of iterations exceeded\)", Jacobi()),
        (Cg(), MixedF32(), 1e-3, r"Cg \(MixedF32\) solve did not converge in 2 iterations \(maximum number of iterations exceeded\)", Jacobi()),
        (Cg(), MixedF32(), 1e-12, r"Cg \(MixedF32\) solve did not converge in \d+ iterations \(iterative refinement stopped after 20 corrections\)", Jacobi()),
        (Cg(), MixedF16(), 1e-12, r"Cg \(MixedF16\) solve did not converge in \d+ iterations \(iterative refinement stopped after 20 corrections\)", Jacobi()),
        (Richardson(), FullPrecision(), 0.0, r"Richardson solve did not converge in 2 iterations \(maximum number of iterations exceeded\)", AMG_PC))
    @test_logs (:warn, msg) match_mode=:any laplace_solve(s, p; itmax=2, rtol=rtol, atol=1e-14, preconditioner=pc)
end

@testset "converged $(nameof(typeof(s))) $(nameof(typeof(p))) solve is silent" for (s, p, rtol, pc) ∈ (
        (Cg(), FullPrecision(), 1e-8, Jacobi()), (Cg(), MixedF32(), 1e-3, Jacobi()), (Cg(), MixedF32(), 1e-8, Jacobi()),
        (Richardson(), FullPrecision(), 1e-8, AMG_PC), (Cg(), FullPrecision(), 1e-8, AMG_PC))
    @test_logs min_level=Warn laplace_solve(s, p; itmax=1000, rtol=rtol, atol=0.0, preconditioner=pc)
end

# rate limited per equation: a second equation still reports after the first one's limit
@testset "unconverged-solve warnings are rate limited per equation" begin
    nmax = XCALibre.Solve.UNCONVERGED_MAXLOG
    sys1, sys2 = (laplace_system(Cg(), FullPrecision(); itmax=2, rtol=0.0, atol=1e-14) for _ ∈ 1:2)
    logger = Test.TestLogger(min_level=Warn)
    with_logger(logger) do
        foreach(_ -> solve_from_zero!(sys1), 1:nmax + 5)
        solve_from_zero!(sys2)
    end
    @test count(l -> occursin("did not converge", string(l.message)), logger.logs) == nmax + 1
end
