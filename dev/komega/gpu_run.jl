#= End-to-end motorBike KOmega run on a chosen backend/assembly. GPU timing only makes sense
   for the whole run: the phase timers wrap asynchronous launches.
     julia --project=<env> dev/komega/gpu_run.jl [iters] [cell|face] [i64|i32] [out]
=#
using XCALibre, JLD2, Printf, Logging, Adapt, KernelAbstractions
iters = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
asmarg= length(ARGS) >= 2 ? ARGS[2] : "cell"
IX    = length(ARGS) >= 3 ? ARGS[3] : "i64"
out   = length(ARGS) >= 4 ? ARGS[4] : "dev/komega/gpu_run.txt"
using CUDA
backend = CUDABackend(); workgroup = 32

BENCH = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS"
meshfile = IX == "i32" ? joinpath(@__DIR__, "mesh_i32.jld2") : joinpath(BENCH, "XCALibre", "mesh.jld2")
mesh_h = load_object(meshfile)
mesh = adapt(backend, mesh_h)

const HAS_FACE = isdefined(XCALibre, :FaceAssembly)
hardware = HAS_FACE ?
    Hardware(backend=backend, workgroup=workgroup,
             assembly = asmarg == "face" ? FaceAssembly() : CellAssembly()) :
    Hardware(backend=backend, workgroup=workgroup)

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

io = open(out*".log","w"); tref = Ref(0.0); finref = Ref{Any}(()); memref = Ref(0)
redirect_stdout(io) do; redirect_stderr(io) do; with_logger(SimpleLogger(io)) do
    init!(); potential_flow!(model, cfg(1); ncorrectors=10); run!(model, cfg(2))  # warm-up + JIT
    KernelAbstractions.synchronize(backend); GC.gc(true); CUDA.reclaim()
    init!(); potential_flow!(model, cfg(iters); ncorrectors=10)
    tref[] = @elapsed begin
        res = run!(model, cfg(iters)); KernelAbstractions.synchronize(backend)
    end
    finref[] = map(last, values(res))
    memref[] = CUDA.memory_status === nothing ? 0 : 0
end; end; end
close(io)
t = tref[]
open(out,"w") do f
    @printf(f, "device=CUDA assembly=%s index=%s iterations=%d  seconds=%.2f  ms/iter=%.2f\n",
        HAS_FACE ? asmarg : "cell(baseline-env)", IX, iters, t, 1000t/iters)
    @printf(f, "final residuals: %s\n", string(finref[]))
    @printf(f, "GPU free/total: %s\n", string(CUDA.available_memory()/2^30) * " / " *
        string(CUDA.total_memory()/2^30) * " GiB")
end
println("done")
