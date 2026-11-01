using XCALibre
using Accessors
using Adapt
using LinearAlgebra
using Test

# One Laplace system solved through every serial linear-solve path: the reported outer residual is
# taken on x₀ before the solve, so it is the same on every path, and each solve meets its rtol.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
parity_mesh = UNV2D_mesh(joinpath(grids_dir, "finer_mesh_laplace.unv"))
const PARITY_RTOL = 1e-3

function parity_solve(solver, preconditioner, precision; backend=CPU(), workgroup=1024)
    mesh = adapt(backend, parity_mesh)
    model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=mesh)
    BCs = assign(region=mesh, (T = [Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall),
        Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
    schemes = (T = Schemes(laplacian = Linear),)
    solvers = (T = SolverSetup(solver=solver, preconditioner=preconditioner, convergence=1e-8, relax=1.0,
        rtol=PARITY_RTOL, atol=0.0, itmax=1000, precision=precision),)
    config = Configuration(solvers=solvers, schemes=schemes, boundaries=BCs,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=backend, workgroup=workgroup))
    T = model.energy.T
    # a non-uniform x₀ so the normalisation Σ|Aᵢᵢxᵢ| is non-trivial
    copyto!(T.values, adapt(backend, [20.0 + 5sin(i) for i ∈ eachindex(parity_mesh.cells)]))
    T_eqn = (
        - Laplacian{schemes.T.laplacian}(model.solid.rDf, T) == - Source(ScalarField(mesh))
    ) → ScalarEquation(T, config.boundaries.T)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)
    @reset T_eqn.preconditioner = set_preconditioner(solvers.T.preconditioner, T_eqn)
    @reset T_eqn.solver = _workspace(solvers.T, T_eqn)
    update_preconditioner!(T_eqn.preconditioner, mesh, config)
    A = Array(parent(XCALibre.ModelFramework._A(T_eqn)))
    b = Array(XCALibre.ModelFramework._b(T_eqn))
    r0 = norm(b - A*Array(T.values))
    res = solve_system!(T_eqn, solvers.T, T, nothing, config)
    res, norm(b - A*Array(T.values))/r0
end

parity_amg = AMG(coarsening=SmoothAggregation(), smoother=AMGJacobi())
parity_paths = (
    ("Cg Jacobi", Cg(), Jacobi(), FullPrecision()),
    ("Cg DILU", Cg(), DILU(), FullPrecision()),
    ("Bicgstab Jacobi", Bicgstab(), Jacobi(), FullPrecision()),
    ("Bicgstab DILU", Bicgstab(), DILU(), FullPrecision()),
    ("AMG", Richardson(), parity_amg, FullPrecision()),
    ("Cg MixedF32", Cg(), Jacobi(), MixedF32()),
    ("Cg MixedF16", Cg(), Jacobi(), MixedF16()))

parity_reference, _ = parity_solve(Cg(), Jacobi(), FullPrecision())

@testset "outer residual parity: $name" for (name, s, P, p) ∈ parity_paths
    res, reduction = parity_solve(s, P, p)
    @test res == parity_reference
    @test reduction ≤ 10PARITY_RTOL # loose: each path measures its own inner norm
end

# GPU reductions sum in a different order, so the residual agrees to rounding only
try
    using CUDA
    if CUDA.functional()
        @testset "outer residual parity: CUDA $name" for (name, s, P, p) ∈ parity_paths[[1, 3, 6]]
            res, reduction = parity_solve(s, P, p; backend=CUDABackend(), workgroup=32)
            @test res ≈ parity_reference rtol=1e-10
            @test reduction ≤ 10PARITY_RTOL
        end
    end
catch err
    @info "Skipping CUDA residual parity test" exception=(err, catch_backtrace())
end
