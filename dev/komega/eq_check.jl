#= Do the two assemblies agree? Compare after 1, 2 and 5 SIMPLE iterations, so that any
disagreement is read before the nonlinear iteration can amplify it. =#
using XCALibre, JLD2, Printf, Logging

BENCH = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS"
mesh = load_object(joinpath(BENCH, "XCALibre", "mesh.jld2"))
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
state() = (sum(model.momentum.U.x.values), sum(model.momentum.p.values),
           sum(model.turbulence.k.values), sum(model.turbulence.omega.values))

io = open("dev/komega/eq_check.txt.log","w"); rows = String[]
redirect_stdout(io) do; redirect_stderr(io) do; with_logger(SimpleLogger(io)) do
    for asm in (CellAssembly(), FaceAssembly())
        init!(); potential_flow!(model, cfg(1,asm); ncorrectors=10); run!(model, cfg(1,asm))
    end
    for n in (1, 2, 5, 20)
        res = Dict{String,Any}()
        for (nm, asm) in (("cell",CellAssembly()), ("face",FaceAssembly()))
            init!(); potential_flow!(model, cfg(n,asm); ncorrectors=10)
            run!(model, cfg(n,asm)); res[nm] = state()
        end
        for (i,lbl) in enumerate(("sumUx","sumP","sumK","sumOmega"))
            c = res["cell"][i]; f = res["face"][i]
            push!(rows, @sprintf("n=%-3d %-9s cell=%.17g  face=%.17g  rel=%.3e", n, lbl, c, f, abs(c-f)/max(abs(c),eps())))
        end
    end
end; end; end
close(io)
open("dev/komega/eq_check.txt","w") do f
    println(f, "threads = $(Threads.nthreads())")
    for r in rows; println(f, r); end
end
