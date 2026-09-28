# shared KOmegaSST BFS cases for the distributed tests: Dirichlet walls, and the wall-function
# set, whose :wall and :top both empty out on some rank at six and eight ranks.

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

# high-Re inlet, so every wall face sits on the log branch and the wall functions are not vacuous
wf_vel = [69.4, 0.0, 0.0]
wf_k = 1.0
wf_w = 1000.0

bfs_sst_wallfn_bcs(mesh) = assign(region=mesh, (
    U = [Dirichlet(:inlet, wf_vel), Zerogradient(:outlet),
         Wall(:wall, [0.0, 0.0, 0.0]), Wall(:top, [0.0, 0.0, 0.0])],
    p = [Neumann(:inlet, 0.0), Dirichlet(:outlet, 0.0), Wall(:wall), Wall(:top)],
    k = [Dirichlet(:inlet, wf_k), Zerogradient(:outlet),
         KWallFunction(:wall), KWallFunction(:top)],
    omega = [Dirichlet(:inlet, wf_w), Zerogradient(:outlet),
         OmegaWallFunction(:wall), OmegaWallFunction(:top)],
    nut = [Dirichlet(:inlet, wf_k/wf_w), Zerogradient(:outlet),
         NutWallFunction(:wall), NutWallFunction(:top)]))

function sst_case(mesh, bcs; iterations, backend=CPU(), walls=(:wall,), init=(0.0, k_in, w_in))
    U0, k0, w0 = init
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=1e-3),
        turbulence = RANS{KOmegaSST}(walls=walls),
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
    initialise!(model.momentum.U, [U0, 0.0, 0.0])
    initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.k, k0)
    initialise!(model.turbulence.omega, w0)
    initialise!(model.turbulence.nut, k0/w0)
    model, config
end

