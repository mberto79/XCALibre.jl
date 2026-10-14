# Steady supersonic flow over a cylinder with the explicit density-based solver:
# global CFL time step against local (per-cell) time stepping, with and without
# the multistage Runge-Kutta scheme (inviscid flow, Mach 1.5). Reports the
# absolute density residual history. Force histories can be requested with
# run!(...; forces=(patches=[:wall], interval=50)) on no-slip walls.
using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "cylinder_d10mm_5mm.unv"), scale=0.001)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

gamma = 1.4; cp = 1005.0; Pr = 0.7; nu = 1e-5
T_inf = 300.0; p_inf = 101325.0
R_gas = cp * (1.0 - 1.0/gamma)
U_inf = 1.5 * sqrt(gamma * R_gas * T_inf)
velocity = [U_inf, 0.0, 0.0]

function run_case(time_stepping, adaptive; iterations=4000)
    model = Physics(
        time = Transient(),
        fluid = Fluid{SupersonicFlow}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
        turbulence = RANS{Laminar}(),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh_dev)
    BCs = assign(region=mesh_dev, (
        U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet),
             Slip(:cylinder), Slip(:top), Slip(:bottom)],
        p = [Dirichlet(:inlet, p_inf), Zerogradient(:outlet), Zerogradient(:cylinder),
             Zerogradient(:top), Zerogradient(:bottom)],
        T = [Dirichlet(:inlet, T_inf), Zerogradient(:outlet), Zerogradient(:cylinder),
             Zerogradient(:top), Zerogradient(:bottom)]))
    schemes = (
        U = Schemes(gradient=Gauss), p = Schemes(gradient=Gauss), T = Schemes(gradient=Gauss),
        flux = HLLC(), time_stepping = time_stepping, reconstruction = MUSCL{VanLeer}())
    runtime = Runtime(iterations=iterations, write_interval=-1, time_step=1e-7, adaptive=adaptive)
    config = Configuration(solvers=(rho=(convergence=1e-15,),), schemes=schemes,
        runtime=runtime, hardware=hardware, boundaries=BCs)
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, p_inf)
    initialise!(model.energy.T, T_inf)
    return run!(model, config; progress=false)
end

cases = [
    "global dt, FEuler, CFL 0.5" => (FEuler(), AdaptiveTimeStepping(maxCo=0.5)),
    "global dt, RK2, CFL 0.5"    => (RK2(),    AdaptiveTimeStepping(maxCo=0.5)),
    "LTS, RK2, CFL 0.5"          => (RK2(),    LocalTimeStepping(maxCo=0.5, startCo=0.1, ramp=200)),
    "LTS, RK3, CFL 1.0"          => (RK3(),    LocalTimeStepping(maxCo=1.0, startCo=0.1, ramp=200)),
]
for (name, (ts, ad)) in cases
    t = @elapsed res = run_case(ts, ad)
    println(rpad(name, 30), " density residual at iterations 500/1000/2000/4000: ",
            round.([res.rho[k] for k in (500, 1000, 2000, 4000)], sigdigits=3),
            "   wall time ", round(t, digits=1), " s")
end
