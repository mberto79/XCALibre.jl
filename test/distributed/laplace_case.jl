# shared Laplace case setup for distributed tests (box = 3D correctness/perf, fine2d = scaling)

box_mesh() = UNV3D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "3d_box_1000x1000x1000mm_10.unv"), scale=0.001)
fine2d_mesh() = UNV2D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "finer_mesh_laplace.unv"))

box_bcs(mesh) = assign(region=mesh, (
    T = [
        Dirichlet(:x_min, 50.0), Zerogradient(:x_max),
        Dirichlet(:y_min, 10.0), Zerogradient(:y_max),
        Zerogradient(:z_min), Zerogradient(:z_max)
    ],))
fine2d_bcs(mesh) = assign(region=mesh, (
    T = [
        Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall),
        Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)
    ],))

function laplace_case(mesh, bcs; iterations=20, convergence=1e-10)
    hardware = Hardware(backend=CPU(), workgroup=64)
    model = Physics(
        time = Steady(),
        solid = Solid{Uniform}(k=1.0),
        energy = Energy{Conduction}(),
        domain = mesh)
    solvers = SolverSetup(
        solver=Cg(), preconditioner=Jacobi(),
        convergence=convergence, relax=1.0, rtol=1e-12, atol=1e-14, itmax=1000)
    schemes = Schemes(laplacian=Linear)
    runtime = Runtime(iterations=iterations, write_interval=-1, time_step=1)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=hardware, boundaries=bcs(mesh))
    initialise!(model.energy.T, 15.0)
    model, config
end

# distributed equation built the way plaplace! builds it (for component-level perf tests)
function build_deqn(dm, model, config)
    (; schemes, boundaries) = config
    (; T) = model.energy
    (; rhocp, rDf) = model.solid
    T_eqn = (
        Time{schemes.time}(rhocp, T)
        - Laplacian{schemes.laplacian}(rDf, T)
        ==
        - Source(ScalarField(dm))
    ) → ScalarEquation(T, boundaries.T)
    XCALibre.Distribute.DistributedEqn(
        T_eqn,
        PETScSolver(T_eqn, dm, config.solvers),
        dm.partition,
        HaloExchange(dm, 1, config.hardware.backend))
end
