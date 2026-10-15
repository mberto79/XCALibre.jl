# Godunov solver with LocalTimeStepping on a GPU (not part of runtests.jl: needs a GPU). Run with an
# environment that has XCALibre and CUDA:
#   julia --project=<env> test/gpu_local_time_stepping.jl
using XCALibre, Test, CUDA

CUDA.functional() || (println("SKIP gpu_local_time_stepping: CUDA not functional"); exit(0))

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
host = UNV2D_mesh(joinpath(grids_dir, "cylinder_d10mm_5mm.unv"), scale=0.001)

gamma = 1.4; cp = 1005.0; Pr = 0.7; nu = 1e-5; T_inf = 300.0; p_inf = 101325.0
a_inf = sqrt(gamma * cp * (1 - 1/gamma) * T_inf)
velocity = [1.2 * a_inf, 0.0, 0.0]

function run_lts(backend, turbulence_dt, time_stepping)
    mesh = adapt(backend, host)
    model = Physics(time=Transient(), fluid=Fluid{SupersonicFlow}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
        turbulence=LES{Smagorinsky}(), energy=Energy{SensibleEnthalpy}(Tref=0.0), domain=mesh)
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Dirichlet(:cylinder, [0.0, 0.0, 0.0]),
             Slip(:top), Slip(:bottom)],
        p = [Dirichlet(:inlet, p_inf), Zerogradient(:outlet), Zerogradient(:cylinder),
             Zerogradient(:top), Zerogradient(:bottom)],
        T = [Dirichlet(:inlet, T_inf), Zerogradient(:outlet), Zerogradient(:cylinder),
             Zerogradient(:top), Zerogradient(:bottom)],
        nut = [Extrapolated(:inlet), Extrapolated(:outlet), Zerogradient(:cylinder),
               Symmetry(:top), Symmetry(:bottom)]))
    schemes = (U=Schemes(gradient=Gauss), p=Schemes(gradient=Gauss), T=Schemes(gradient=Gauss),
        flux=HLLC(), time_stepping=time_stepping, reconstruction=MUSCL{VanLeer}())
    lts = LocalTimeStepping(maxCo=0.5, startCo=0.1, ramp=5, turbulence_dt=turbulence_dt)
    @test isbitstype(typeof(lts))
    config = Configuration(solvers=(rho=(convergence=1e-15,),), schemes=schemes,
        runtime=Runtime(iterations=20, write_interval=-1, time_step=1e-7, adaptive=lts),
        hardware=Hardware(backend=backend, workgroup=backend isa CPU ? 1024 : 32), boundaries=BCs)
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, p_inf)
    initialise!(model.energy.T, T_inf)
    res = run!(model, config; progress=false)
    return res, Array(model.fluid.rho.values)
end

@testset "godunov! with LocalTimeStepping on GPU" begin
    for dtrule in (:min, :mean, :max), ts in (RK2(), RK3())
        @testset "turbulence_dt=$dtrule $(typeof(ts))" begin
            res, rho = run_lts(CUDABackend(), dtrule, ts)
            @test all(isfinite, res.rho)
            @test all(isfinite, rho) && all(>(0), rho)
        end
    end
    @testset "CPU/GPU agreement" begin
        rc, rhoc = run_lts(CPU(), :min, RK2())
        rg, rhog = run_lts(CUDABackend(), :min, RK2())
        @test maximum(abs.(rhog .- rhoc)) < 1e-8 * maximum(rhoc)
    end
end
