# shared incompressible cases for Phase 5 gates (BFS + lid-driven cavity, 2D laminar).
# Fixed iteration budget (convergence=1e-15 unreachable) + tight inner tolerances make the
# outer trajectory solver-agnostic (Krylov vs PETSc KSP), so fields compare at 1e-6.

bfs_mesh() = UNV2D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "backwardFacingStep_10mm.unv"), scale=0.001)
cavity_mesh() = UNV2D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "quad40.unv"), scale=0.001)

bfs_bcs(mesh) = assign(region=mesh, (
    U = [Dirichlet(:inlet, [0.5, 0.0, 0.0]), Extrapolated(:outlet),
         Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
    p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0),
         Extrapolated(:wall), Symmetry(:top)]))

# quad40 patches: inlet/outlet/top/bottom; lid = top
cavity_bcs(mesh) = assign(region=mesh, (
    U = [Wall(:inlet, [0.0, 0.0, 0.0]), Wall(:outlet, [0.0, 0.0, 0.0]),
         Dirichlet(:top, [1.0, 0.0, 0.0]), Wall(:bottom, [0.0, 0.0, 0.0])],
    p = [Zerogradient(:inlet), Zerogradient(:outlet),
         Zerogradient(:top), Zerogradient(:bottom)]))

function incompressible_case(mesh, bcs; iterations, time=Steady(), time_step=1, backend=CPU())
    model = Physics(
        time = time,
        fluid = Fluid{Incompressible}(nu=1e-3),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh)
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.8, rtol=1e-8, atol=1e-12, itmax=2000),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.2, rtol=1e-8, atol=1e-12, itmax=2000))
    schemes = (U=Schemes(divergence=Linear, time=(time isa Steady ? SteadyState : Euler)),
               p=Schemes())
    runtime = Runtime(iterations=iterations, time_step=time_step, write_interval=-1)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=backend, workgroup=64), boundaries=bcs(mesh))
    initialise!(model.momentum.U, [0.0, 0.0, 0.0])
    initialise!(model.momentum.p, 0.0)
    model, config
end

# owned-cell max |Δ| against a serial reference indexed by original global id
function field_errors(dm, model_d, Us_x, Us_y, ps)
    n = dm.partition.n_owned
    orig = dm.orig_cells
    ux, uy = Array(model_d.momentum.U.x.values), Array(model_d.momentum.U.y.values)
    pv = Array(model_d.momentum.p.values)
    dux = maximum(abs.(ux[1:n] .- Us_x[orig[1:n]]); init=0.0)
    duy = maximum(abs.(uy[1:n] .- Us_y[orig[1:n]]); init=0.0)
    dp = maximum(abs.(pv[1:n] .- ps[orig[1:n]]); init=0.0)
    dux, duy, dp
end
