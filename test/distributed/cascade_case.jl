# shared 3D periodic cascade case for the distributed tests: the only 3D mesh in the suite with
# periodic patches, so it covers the 3D branches and `construct_periodic` in one run.
cascade_mesh() = UNV3D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "cascade_3D_periodic_4mm.unv"), scale=0.001)

cascade_vel = [0.25, 0.0, 0.0]

cascade_bcs(mesh) = begin
    periodic = construct_periodic(mesh, CPU(), :top, :bottom)
    assign(region=mesh, (
        U = [Dirichlet(:inlet, cascade_vel), Extrapolated(:outlet),
             Wall(:plate, [0.0, 0.0, 0.0]), Extrapolated(:side1), Extrapolated(:side2),
             periodic...],
        p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Extrapolated(:plate),
             Extrapolated(:side1), Extrapolated(:side2), periodic...]))
end

function cascade_case(mesh; iterations, backend=CPU())
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=1e-3),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh)
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.8, rtol=1e-8, atol=1e-12, itmax=2000),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.2, rtol=1e-8, atol=1e-12, itmax=2000))
    schemes = (U=Schemes(divergence=Linear), p=Schemes())
    runtime = Runtime(iterations=iterations, time_step=1, write_interval=-1)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=backend, workgroup=64), boundaries=cascade_bcs(mesh))
    initialise!(model.momentum.U, cascade_vel)
    initialise!(model.momentum.p, 0.0)
    model, config
end
