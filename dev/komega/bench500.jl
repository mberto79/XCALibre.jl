#= The benchmark as the benchmark runs it: 500 iterations, one timed run, compared against
OpenFOAM's own numbers for the same case (1 core 238.87 s, 8 cores 81.66 s).
  julia --project=dev/komega -t N dev/komega/bench500.jl [iterations] [out]
=#
using XCALibre, JLD2, Printf, ThreadPinning, Logging

iters = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
out   = length(ARGS) >= 2 ? ARGS[2] : "dev/komega/bench500.txt"
BENCH = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS"
mesh = load_object(joinpath(BENCH, "XCALibre", "mesh.jld2"))
pinthreads(:cores)
hardware = Hardware(backend=CPU(static=true), workgroup=AutoTune())  # CellAssembly default

velocity = [20.0,0.0,0.0]; noSlip = [0.0,0.0,0.0]
nu = 1.5e-5; k_inlet = 0.24; omega_inlet = 1.78; nut_inlet = k_inlet/omega_inlet
model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=nu),
    turbulence=RANS{KOmega}(), energy=Energy{Isothermal}(), domain=mesh)
BCs = assign(region = mesh, (
    U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Wall(:lowerWall, velocity),
         Wall(:motorBike, noSlip), Slip(:upperWall), Slip(:frontAndBack)],
    p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:lowerWall),
         Wall(:motorBike), Slip(:upperWall), Slip(:frontAndBack)],
    k = [Dirichlet(:inlet, k_inlet), Zerogradient(:outlet), KWallFunction(:lowerWall),
         KWallFunction(:motorBike), Slip(:upperWall), Slip(:frontAndBack)],
    omega = [Dirichlet(:inlet, omega_inlet), Zerogradient(:outlet), OmegaWallFunction(:lowerWall),
         OmegaWallFunction(:motorBike), Slip(:upperWall), Slip(:frontAndBack)],
    nut = [Dirichlet(:inlet, nut_inlet), Zerogradient(:outlet), NutWallFunction(:lowerWall),
         NutWallFunction(:motorBike), Slip(:upperWall), Slip(:frontAndBack)]))
mk(sol, rx) = SolverSetup(solver=sol, preconditioner=Jacobi(), convergence=1e-14, relax=rx, rtol=0.1, itmax=1000)
solvers = (U=mk(Bicgstab(),0.7), p=mk(Cg(),0.3), k=mk(Bicgstab(),0.3), omega=mk(Bicgstab(),0.3))
schemes = (U=Schemes(time=SteadyState,divergence=LUST,gradient=Gauss),
           p=Schemes(time=SteadyState,gradient=Gauss),
           k=Schemes(time=SteadyState,divergence=Upwind,gradient=Gauss),
           omega=Schemes(time=SteadyState,divergence=Upwind,gradient=Gauss))
init!() = (initialise!(model.momentum.U, velocity); initialise!(model.momentum.p, 0.0);
    initialise!(model.turbulence.k, k_inlet); initialise!(model.turbulence.omega, omega_inlet);
    initialise!(model.turbulence.nut, nut_inlet))
cfg(n) = Configuration(solvers=solvers, schemes=schemes,
    runtime=Runtime(iterations=n, write_interval=-1, time_step=1), hardware=hardware, boundaries=BCs)

io = open(out*".log","w"); tref = Ref(0.0); finref = Ref{Any}(())
redirect_stdout(io) do; redirect_stderr(io) do; with_logger(SimpleLogger(io)) do
    init!(); potential_flow!(model, cfg(1); ncorrectors=10); run!(model, cfg(1))  # warm-up
    GC.gc(true)
    init!(); potential_flow!(model, cfg(iters); ncorrectors=10)
    tref[] = @elapsed res = run!(model, cfg(iters))
    finref[] = map(last, values(res))
end; end; end
close(io)
t = tref[]
open(out,"w") do f
    @printf(f, "threads=%d  iterations=%d  seconds=%.2f  ms/iter=%.1f\n",
        Threads.nthreads(), iters, t, 1000t/iters)
    @printf(f, "final residuals: %s\n", string(finref[]))
    @printf(f, "OpenFOAM same case: 1 core 238.87 s, 2 cores 170.52, 6 cores 90.89, 8 cores 81.66\n")
end
