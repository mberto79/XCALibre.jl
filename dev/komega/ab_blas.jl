# A/B the BLAS thread count inside one process, alternating, so thermal drift and machine
# noise hit both arms equally. Everything else is identical.
using XCALibre, JLD2, Printf, ThreadPinning, Logging, LinearAlgebra

reps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
iters = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 15
out = length(ARGS) >= 3 ? ARGS[3] : "dev/komega/ab_blas.txt"

BENCH = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS"
mesh = load_object(joinpath(BENCH, "XCALibre", "mesh.jld2"))
pinthreads(:cores)
hardware = Hardware(backend=CPU(static=true), workgroup=AutoTune())

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

io = open(out*".log","w"); res = Dict{Int,Vector{Float64}}()
redirect_stdout(io) do; redirect_stderr(io) do; with_logger(SimpleLogger(io)) do
    BLAS.set_num_threads(1); init!(); potential_flow!(model, cfg(1); ncorrectors=10); run!(model, cfg(1))
    for r in 1:reps, nb in (1, Threads.nthreads())
        BLAS.set_num_threads(nb); GC.gc(true)
        init!(); potential_flow!(model, cfg(iters); ncorrectors=10)
        t = @elapsed run!(model, cfg(iters))
        push!(get!(res, nb, Float64[]), 1000t/iters)
    end
end; end; end
close(io)
open(out,"w") do f
    println(f, "julia threads = $(Threads.nthreads()), iterations = $iters, reps = $reps")
    for nb in sort(collect(keys(res)))
        v = res[nb]
        @printf(f, "BLAS threads %2d : %s  | min %.1f ms/iter  median %.1f\n", nb,
            join([@sprintf("%.1f", x) for x in v], " "), minimum(v), sort(v)[cld(length(v),2)])
    end
end
