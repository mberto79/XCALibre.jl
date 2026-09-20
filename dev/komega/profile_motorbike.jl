#=
motorBike KOmega profiling harness (branch HM/KOmega-profiling).

  julia --project -t N dev/komega/profile_motorbike.jl <iterations> <outfile> [profile] [i32]

Mesh is loaded from the benchmark directory; nothing is copied into the repo.
All solver output goes to <outfile>.log; only <outfile> is meant to be read.
=#

using XCALibre
using JLD2
using Profile
using Printf
using ThreadPinning
using Logging
using LinearAlgebra

iterations = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
outfile    = length(ARGS) >= 2 ? ARGS[2] : "dev/komega/out.txt"
do_profile = "profile" in ARGS
IX         = "i32" in ARGS ? "i32" : "i64"

BENCH = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS"
mesh = load_object(IX == "i32" ? joinpath(@__DIR__, "mesh_i32.jld2") :
                                 joinpath(BENCH, "XCALibre", "mesh.jld2"))

pinthreads(:cores)
backend = CPU(static=true)
workgroup = AutoTune()
activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)

velocity = [20.0, 0.0, 0.0]; noSlip = [0.0, 0.0, 0.0]
nu = 1.5e-5; k_inlet = 0.24; omega_inlet = 1.78; nut_inlet = k_inlet/omega_inlet

model = Physics(
    time = Steady(), fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(), energy = Energy{Isothermal}(), domain = mesh)

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

mk(relax; rtol=0.1, itmax=1000) = SolverSetup(
    solver=relax[1], preconditioner=Jacobi(), convergence=1e-14,
    relax=relax[2], rtol=rtol, itmax=itmax)

solvers = (
    U     = mk((Bicgstab(), 0.7)),
    p     = mk((Cg(), 0.3)),
    k     = mk((Bicgstab(), 0.3)),
    omega = mk((Bicgstab(), 0.3)))

schemes = (
    U     = Schemes(time=SteadyState, divergence=LUST,   gradient=Gauss),
    p     = Schemes(time=SteadyState,                    gradient=Gauss),
    k     = Schemes(time=SteadyState, divergence=Upwind, gradient=Gauss),
    omega = Schemes(time=SteadyState, divergence=Upwind, gradient=Gauss))

init!() = begin
    initialise!(model.momentum.U, velocity); initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.k, k_inlet); initialise!(model.turbulence.omega, omega_inlet)
    initialise!(model.turbulence.nut, nut_inlet)
end

cfg(n) = Configuration(solvers=solvers, schemes=schemes,
    runtime=Runtime(iterations=n, write_interval=-1, time_step=1),
    hardware=hardware, boundaries=BCs)

log_io = open(outfile * ".log", "w")
summary = IOBuffer()

redirect_stdout(log_io) do
redirect_stderr(log_io) do
with_logger(SimpleLogger(log_io)) do

    # warm-up: compile everything outside the timer
    init!(); potential_flow!(model, cfg(1); ncorrectors=10); run!(model, cfg(1))

    GC.gc(true)
    init!(); potential_flow!(model, cfg(iterations); ncorrectors=10)
    t = @elapsed res = run!(model, cfg(iterations))
    @printf(summary, "threads=%d  blas=%d  index=%s  iterations=%d  cells=%d\n",
            Threads.nthreads(), BLAS.get_num_threads(), IX, iterations, length(mesh.cells))
    @printf(summary, "wall=%.3f s   per-iteration=%.1f ms\n", t, 1000t/iterations)
    for kk in keys(res)
        @printf(summary, "  res[%s] = %.17g\n", kk, last(res[kk]))
    end
    @printf(summary, "  k=%.17g omega=%.17g nut=%.17g\n",
        sum(model.turbulence.k.values), sum(model.turbulence.omega.values),
        sum(model.turbulence.nut.values))

    # phase timers: second run, instrumented
    GC.gc(true)
    init!(); potential_flow!(model, cfg(iterations); ncorrectors=10)
    xcprof_reset!(); XCPROF[] = true
    t2 = @elapsed run!(model, cfg(iterations))
    XCPROF[] = false
    @printf(summary, "instrumented wall=%.3f s  per-iteration=%.1f ms\n\n", t2, 1000t2/iterations)
    xcprof_report(summary; iterations=iterations, total=t2)

    if do_profile
        GC.gc(true)
        init!(); potential_flow!(model, cfg(iterations); ncorrectors=10)
        Profile.clear(); Profile.init(n=20_000_000, delay=0.0005)
        Profile.@profile run!(model, cfg(iterations))
        println(summary, "\n===== FLAT PROFILE (self time, C=true) =====")
        Profile.print(IOContext(summary, :displaysize=>(24,220));
                      C=true, format=:flat, sortedby=:overhead, mincount=100)
    end
end
end
end
close(log_io)

open(outfile, "w") do io; write(io, take!(summary)); end
