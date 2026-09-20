#= End-to-end GPU wall time for all four combinations of assembly and index type, in one
   process because each new process pays the full GPU kernel compilation.
     julia --project=dev/komega dev/komega/gpu_run4.jl [iters] [out]
=#
using XCALibre, JLD2, Printf, Logging, Adapt, KernelAbstractions, CUDA
iters = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
out   = length(ARGS) >= 2 ? ARGS[2] : "dev/komega/gpu_run4.txt"
backend = CUDABackend(); workgroup = 32
sync() = KernelAbstractions.synchronize(backend)
BENCH = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS"
meshpath(ix) = ix == "i32" ? joinpath(@__DIR__, "mesh_i32.jld2") :
                             joinpath(BENCH, "XCALibre", "mesh.jld2")

velocity = [20.0,0.0,0.0]; noSlip = [0.0,0.0,0.0]
nu = 1.5e-5; k_inlet = 0.24; omega_inlet = 1.78; nut_inlet = k_inlet/omega_inlet
mk(sol, rx) = SolverSetup(solver=sol, preconditioner=Jacobi(), convergence=1e-14, relax=rx, rtol=0.1, itmax=1000)
solvers = (U=mk(Bicgstab(),0.7), p=mk(Cg(),0.3), k=mk(Bicgstab(),0.3), omega=mk(Bicgstab(),0.3))
schemes = (U=Schemes(time=SteadyState,divergence=LUST,gradient=Gauss),
           p=Schemes(time=SteadyState,gradient=Gauss),
           k=Schemes(time=SteadyState,divergence=Upwind,gradient=Gauss),
           omega=Schemes(time=SteadyState,divergence=Upwind,gradient=Gauss))

rows = String[]
io = open(out*".log","w")
redirect_stdout(io) do; redirect_stderr(io) do; with_logger(SimpleLogger(io)) do
for ix in ("i64", "i32"), (label, asm) in (("cell", CellAssembly()), ("face", FaceAssembly()))
    mesh = adapt(backend, load_object(meshpath(ix)))
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
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=nu),
        turbulence=RANS{KOmega}(), energy=Energy{Isothermal}(), domain=mesh)
    hardware = Hardware(backend=backend, workgroup=workgroup, assembly=asm)
    cfg(n) = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=n, write_interval=-1, time_step=1),
        hardware=hardware, boundaries=BCs)
    init!() = (initialise!(model.momentum.U, velocity); initialise!(model.momentum.p, 0.0);
        initialise!(model.turbulence.k, k_inlet); initialise!(model.turbulence.omega, omega_inlet);
        initialise!(model.turbulence.nut, nut_inlet))

    init!(); potential_flow!(model, cfg(2); ncorrectors=10); run!(model, cfg(2)) # warm-up + JIT
    sync(); GC.gc(true); CUDA.reclaim()
    init!(); potential_flow!(model, cfg(iters); ncorrectors=10)
    t = @elapsed (res = run!(model, cfg(iters)); sync())
    push!(rows, @sprintf("%-4s %-5s %7.2f s   %7.2f ms/iter   final=%s", ix, label, t,
        1000t/iters, string(map(x->round(last(x), sigdigits=5), values(res)))))
    model = nothing; mesh = nothing; BCs = nothing
    GC.gc(true); CUDA.reclaim()
end
end; end; end
close(io)
open(out, "w") do f
    @printf(f, "CUDA RTX 4070 Laptop, motorBike KOmega, %d iterations, timed with @elapsed run! + synchronize\n", iters)
    for r in rows; println(f, r); end
    println(f, "\nCPU reference, 500 iterations: t8 i64 194.0 ms/iter, t8 i32 153.4, t1 i64 356.8, t1 i32 328.3")
    println(f, "(iteration counts differ, so per-iteration figures are not directly comparable to the CPU ones)")
end
println("done")
