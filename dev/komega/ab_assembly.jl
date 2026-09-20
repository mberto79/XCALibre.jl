#=
Face-based vs cell-based assembly: equivalence and speed, in one process with the arms
alternating so machine drift hits both equally.

  julia --project=dev/komega -t N dev/komega/ab_assembly.jl <reps> <iters> <out>
=#
using XCALibre, JLD2, Printf, ThreadPinning, Logging, LinearAlgebra

reps  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
iters = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
out   = length(ARGS) >= 3 ? ARGS[3] : "dev/komega/ab_assembly.txt"

BENCH = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS"
mesh = load_object(joinpath(BENCH, "XCALibre", "mesh.jld2"))
pinthreads(:cores)

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
cfg(n, asm) = Configuration(solvers=solvers, schemes=schemes,
    runtime=Runtime(iterations=n, write_interval=-1, time_step=1),
    hardware=Hardware(backend=CPU(static=true), workgroup=AutoTune(), assembly=asm),
    boundaries=BCs)

state() = (sum(model.momentum.U.x.values), sum(model.momentum.U.y.values),
           sum(model.momentum.p.values), sum(model.turbulence.k.values),
           sum(model.turbulence.omega.values), sum(model.turbulence.nut.values))

io = open(out*".log","w")
times = Dict{String,Vector{Float64}}(); finals = Dict{String,Any}(); phases = Dict{String,Any}()

function write_report()
    open(out,"w") do f
        println(f, "julia threads = $(Threads.nthreads()), iterations = $iters, reps = $reps")
        for nm in ("cell","face")
            haskey(times, nm) || continue
            v = times[nm]
            @printf(f, "%-5s : %s  | min %.1f ms/iter  median %.1f\n", nm,
                join([@sprintf("%.1f", x) for x in v], " "), minimum(v), sort(v)[cld(length(v),2)])
        end
        if haskey(finals,"cell") && haskey(finals,"face")
            println(f, "\nequivalence (cell vs face, relative):")
            for (i, lbl) in enumerate(("Ux_res","Uy_res","Uz_res","p_res","sumUx","sumUy","sumP","sumK","sumOmega","sumNut"))
                c = finals["cell"][i]; fa = finals["face"][i]
                @printf(f, "  %-9s cell=%.17g  face=%.17g  rel=%.3e\n", lbl, c, fa, abs(c-fa)/max(abs(c), eps()))
            end
        end
        for nm in ("cell","face")
            haskey(phases, nm) && println(f, "\n===== $nm =====\n", phases[nm])
        end
    end
end
redirect_stdout(io) do; redirect_stderr(io) do; with_logger(SimpleLogger(io)) do
    for asm in (CellAssembly(), FaceAssembly())
        init!(); potential_flow!(model, cfg(1, asm); ncorrectors=10); run!(model, cfg(1, asm))
    end
    for r in 1:reps, (nm, asm) in (("cell", CellAssembly()), ("face", FaceAssembly()))
        GC.gc(true)
        init!(); potential_flow!(model, cfg(iters, asm); ncorrectors=10)
        xcprof_reset!(); XCPROF[] = true
        t = @elapsed res = run!(model, cfg(iters, asm))
        XCPROF[] = false
        push!(get!(times, nm, Float64[]), 1000t/iters)
        finals[nm] = (map(last, values(res))..., state()...)
        if r == 1
            b = IOBuffer(); xcprof_report(b; iterations=iters, total=t); phases[nm] = String(take!(b))
        end
        write_report()
    end
end; end; end
close(io)

