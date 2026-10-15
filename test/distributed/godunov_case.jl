# shared density-based (godunov!) case for the distributed tests: Mach 1.2 flow past a 2D
# cylinder, laminar or KOmegaSST with Dirichlet walls (no wall functions) so the serial and
# distributed runs match to round-off.

cylinder_mesh() = UNV2D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "cylinder_d10mm_5mm.unv"), scale=0.001)

const g_gamma = 1.4
const g_cp = 1005.0
const g_T = 300.0
const g_p = 101325.0
const g_U = 1.2 * sqrt(g_gamma * g_cp * (1.0 - 1.0/g_gamma) * g_T)
const g_k = 1.5 * (0.01 * g_U)^2
const g_w = 1e5

function godunov_bcs(mesh, sst)
    U = [Dirichlet(:inlet, [g_U, 0.0, 0.0]), Zerogradient(:outlet),
         Wall(:cylinder, [0.0, 0.0, 0.0]), Slip(:top), Slip(:bottom)]
    p = [Dirichlet(:inlet, g_p), Zerogradient(:outlet), Zerogradient(:cylinder),
         Zerogradient(:top), Zerogradient(:bottom)]
    T = [Dirichlet(:inlet, g_T), Zerogradient(:outlet), Zerogradient(:cylinder),
         Zerogradient(:top), Zerogradient(:bottom)]
    sst || return assign(region=mesh, (; U, p, T))
    k = [Dirichlet(:inlet, g_k), Zerogradient(:outlet), Dirichlet(:cylinder, 0.0),
         Zerogradient(:top), Zerogradient(:bottom)]
    omega = [Dirichlet(:inlet, g_w), Zerogradient(:outlet), Dirichlet(:cylinder, 1e6),
         Zerogradient(:top), Zerogradient(:bottom)]
    nut = [Dirichlet(:inlet, g_k/g_w), Zerogradient(:outlet), Dirichlet(:cylinder, 0.0),
         Zerogradient(:top), Zerogradient(:bottom)]
    assign(region=mesh, (; U, p, T, k, omega, nut))
end

function godunov_case(mesh; sst=false, iterations=30, flux=HLLC(), reconstruction=MUSCL{VanLeer}(),
        time_stepping=RK2(), adaptive=nothing, gradient=Gauss, backend=CPU())
    model = Physics(
        time = Transient(),
        fluid = Fluid{SupersonicFlow}(nu=1e-5, cp=g_cp, gamma=g_gamma, Pr=0.7),
        turbulence = sst ? RANS{KOmegaSST}(walls=(:cylinder,)) : RANS{Laminar}(),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh)
    tight() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-15, relax=1.0, rtol=1e-13, atol=1e-16, itmax=2000)
    solvers = sst ? (rho=(convergence=1e-15,), k=tight(), omega=tight()) : (rho=(convergence=1e-15,),)
    schemes = (U=Schemes(gradient=gradient), p=Schemes(gradient=gradient),
        T=Schemes(gradient=gradient), k=Schemes(), omega=Schemes(),
        flux=flux, reconstruction=reconstruction, time_stepping=time_stepping)
    runtime = adaptive === nothing ?
        Runtime(iterations=iterations, write_interval=-1, time_step=1e-7) :
        Runtime(iterations=iterations, write_interval=-1, time_step=1e-7, adaptive=adaptive)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=backend, workgroup=64), boundaries=godunov_bcs(mesh, sst))
    initialise!(model.momentum.U, [g_U, 0.0, 0.0])
    initialise!(model.momentum.p, g_p)
    initialise!(model.energy.T, g_T)
    if sst
        initialise!(model.turbulence.k, g_k)
        initialise!(model.turbulence.omega, g_w)
        initialise!(model.turbulence.nut, g_k/g_w)
    end
    model, config
end

# cell fields compared between the serial and distributed runs
godunov_fields(model) = begin
    f = [model.momentum.U.x, model.momentum.U.y, model.momentum.p, model.energy.T, model.fluid.rho]
    model.turbulence isa KOmegaSST &&
        append!(f, [model.turbulence.k, model.turbulence.omega, model.turbulence.nut])
    f
end

# 3D: Mach 1.2 stream through a box, no-slip floor (wall distance, SST), slip elsewhere
box_mesh() = UNV3D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "3d_box_1000x1000x1000mm_5.unv"), scale=0.001)

function box_bcs(mesh)
    slip = (:y_max, :z_min, :z_max)
    zg(names...) = [Zerogradient(s) for s ∈ names]
    assign(region=mesh, (
        U = [Dirichlet(:x_min, [g_U, 0.0, 0.0]), Zerogradient(:x_max),
             Wall(:y_min, [0.0, 0.0, 0.0]), [Slip(s) for s ∈ slip]...],
        p = [Dirichlet(:x_min, g_p), zg(:x_max, :y_min, slip...)...],
        T = [Dirichlet(:x_min, g_T), zg(:x_max, :y_min, slip...)...],
        k = [Dirichlet(:x_min, g_k), Zerogradient(:x_max), Dirichlet(:y_min, 0.0), zg(slip...)...],
        omega = [Dirichlet(:x_min, g_w), Zerogradient(:x_max), Dirichlet(:y_min, 1e6), zg(slip...)...],
        nut = [Dirichlet(:x_min, g_k/g_w), Zerogradient(:x_max), Dirichlet(:y_min, 0.0), zg(slip...)...]))
end

function box_case(mesh; iterations=10, write_interval=-1)
    model = Physics(
        time = Transient(),
        fluid = Fluid{SupersonicFlow}(nu=1e-5, cp=g_cp, gamma=g_gamma, Pr=0.7),
        turbulence = RANS{KOmegaSST}(walls=(:y_min,)),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh)
    tight() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-15, relax=1.0, rtol=1e-13, atol=1e-16, itmax=2000)
    schemes = (U=Schemes(), p=Schemes(), T=Schemes(), k=Schemes(), omega=Schemes(),
        flux=Rusanov(), reconstruction=MUSCL{MinMod}(), time_stepping=RK2())
    runtime = Runtime(iterations=iterations, write_interval=write_interval, time_step=1e-7,
        adaptive=LocalTimeStepping(maxCo=0.5))
    config = Configuration(solvers=(rho=(convergence=1e-15,), k=tight(), omega=tight()),
        schemes=schemes, runtime=runtime, hardware=Hardware(backend=CPU(), workgroup=64),
        boundaries=box_bcs(mesh))
    initialise!(model.momentum.U, [g_U, 0.0, 0.0])
    initialise!(model.momentum.p, g_p)
    initialise!(model.energy.T, g_T)
    initialise!(model.turbulence.k, g_k)
    initialise!(model.turbulence.omega, g_w)
    initialise!(model.turbulence.nut, g_k/g_w)
    model, config
end
