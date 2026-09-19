# shared KOmegaSST BFS case (Dirichlet walls, no wall functions) for the distributed tests

bfs_mesh() = UNV2D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "backwardFacingStep_10mm.unv"), scale=0.001)

k_in = 0.01
w_in = 100.0

bfs_sst_bcs(mesh) = assign(region=mesh, (
    U = [Dirichlet(:inlet, [0.5, 0.0, 0.0]), Extrapolated(:outlet),
         Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
    p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0),
         Extrapolated(:wall), Symmetry(:top)],
    k = [Dirichlet(:inlet, k_in), Zerogradient(:outlet),
         Dirichlet(:wall, 1e-12), Symmetry(:top)],
    omega = [Dirichlet(:inlet, w_in), Zerogradient(:outlet),
         Dirichlet(:wall, w_in), Symmetry(:top)],
    nut = [Dirichlet(:inlet, k_in/w_in), Extrapolated(:outlet),
         Dirichlet(:wall, 0.0), Symmetry(:top)]))

function sst_case(mesh, bcs; iterations, backend=CPU())
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=1e-3),
        turbulence = RANS{KOmegaSST}(walls=(:wall,)),
        energy = Energy{Isothermal}(),
        domain = mesh)
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.7, rtol=1e-8, atol=1e-12, itmax=2000),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.3, rtol=1e-8, atol=1e-12, itmax=2000),
        k = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.6, rtol=1e-8, atol=1e-12, itmax=2000),
        omega = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.6, rtol=1e-8, atol=1e-12, itmax=2000),
        y = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
            convergence=1e-15, relax=1.0, rtol=1e-8, atol=1e-12, itmax=2000))
    schemes = (U=Schemes(divergence=Upwind), p=Schemes(),
               k=Schemes(divergence=Upwind), omega=Schemes(divergence=Upwind), y=Schemes())
    runtime = Runtime(iterations=iterations, time_step=1, write_interval=-1)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=backend, workgroup=64), boundaries=bcs(mesh))
    initialise!(model.momentum.U, [0.0, 0.0, 0.0])
    initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.k, k_in)
    initialise!(model.turbulence.omega, w_in)
    initialise!(model.turbulence.nut, k_in/w_in)
    model, config
end

