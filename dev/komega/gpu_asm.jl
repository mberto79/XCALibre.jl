#= Isolated assembly cost, plus an optional short solve. KA kernels are asynchronous, so every
   timed region ends with an explicit synchronize; the solver's phase timers wrap launches and
   cannot be trusted on GPU. One process does every variant because each new process pays the
   full GPU kernel compilation (~15 min on this case).
     julia --project=<env> dev/komega/gpu_asm.jl [gpu|cpu] [i64|i32|both] [reps] [out] [run]
   Runs in the current env and in the pre-gDiff baseline env (FaceAssembly guarded).
=#
using XCALibre, JLD2, Printf, Random, KernelAbstractions, Adapt, Logging
const DEV = length(ARGS) >= 1 ? ARGS[1] : "gpu"
const IX  = length(ARGS) >= 2 ? ARGS[2] : "i64"
const REPS= length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
const OUT = length(ARGS) >= 4 ? ARGS[4] : "dev/komega/gpu_asm.txt"
const RUN = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
if DEV == "gpu"
    using CUDA
    backend = CUDABackend(); workgroup = 32
else
    backend = CPU(static=true); workgroup = AutoTune()
end
const HAS_FACE = isdefined(XCALibre, :FaceAssembly)
sync() = KernelAbstractions.synchronize(backend)
BENCH = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS"
meshpath(ix) = ix == "i32" ? joinpath(@__DIR__, "mesh_i32.jld2") :
                             joinpath(BENCH, "XCALibre", "mesh.jld2")

velocity = [20.0,0.0,0.0]; noSlip = [0.0,0.0,0.0]
nu = 1.5e-5; k_inlet = 0.24; omega_inlet = 1.78; nut_inlet = k_inlet/omega_inlet
mkBCs(mesh) = assign(region = mesh, (
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
mkhw(asm) = HAS_FACE && asm !== nothing ?
    Hardware(backend=backend, workgroup=workgroup, assembly=asm) :
    Hardware(backend=backend, workgroup=workgroup)

# Three warm-up launches, not one: kernel compilation and the first device allocations must be
# out of the way before the timed reps. The arms are then interleaved rep by rep so a laptop
# GPU's clock drift hits both equally.
function bench_all(fs)
    for _ in 1:3, (_, f) in fs; f(); end
    sync()
    best = Dict(l => Inf for (l, _) in fs)
    for _ in 1:REPS, (l, f) in fs
        t = @elapsed (f(); sync())
        best[l] = min(best[l], t)
    end
    best
end

rows = String[]
for ix in (IX == "both" ? ("i64","i32") : (IX,))
    mesh = adapt(backend, load_object(meshpath(ix)))
    BCs = mkBCs(mesh)
    cfg(asm) = Configuration(
        solvers = (U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),
                   p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),
                   k = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)),
        schemes = (U = Schemes(time=SteadyState, divergence=LUST, laplacian=Linear, gradient=Gauss),
                   p = Schemes(time=SteadyState, laplacian=Linear, gradient=Gauss),
                   k = Schemes(time=SteadyState, divergence=Upwind, laplacian=Linear, gradient=Gauss)),
        runtime = Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware = mkhw(asm), boundaries = BCs)

    Random.seed!(7)
    nc = length(mesh.cells); nf = length(mesh.faces)
    phi = ScalarField(mesh); U = VectorField(mesh)
    mueff = FaceScalarField(mesh); mdotf = FaceScalarField(mesh)
    Dkf = ScalarField(mesh); Pk = ScalarField(mesh); gradp = VectorField(mesh)
    rho = ConstantScalar(1.0)
    copyto!(phi.values, rand(nc)); copyto!(Dkf.values, rand(nc)); copyto!(Pk.values, rand(nc))
    copyto!(mueff.values, 1e-5 .+ rand(nf)); copyto!(mdotf.values, randn(nf))
    for c in (U.x, U.y, U.z, gradp.x, gradp.y, gradp.z); copyto!(c.values, rand(nc)); end

    k_eqn = (Time{SteadyState}(rho, phi) + Divergence{Upwind}(mdotf, phi)
             - Laplacian{Linear}(mueff, phi) + Si(Dkf, phi) == Source(Pk)) → ScalarEquation(phi, BCs.k)
    p_eqn = (- Laplacian{Linear}(mueff, phi) == - Source(Pk)) → ScalarEquation(phi, BCs.p)
    U_eqn = (Time{SteadyState}(U) + Divergence{LUST}(mdotf, U)
             - Laplacian{Linear}(mueff, U) == - Source(gradp)) → VectorEquation(U, BCs.U)

    asms = HAS_FACE ? [("cell", CellAssembly()), ("face", FaceAssembly())] : [("cell", nothing)]
    fs = Pair{String,Function}[]
    for (label, asm) in asms
        c = cfg(asm)
        push!(fs, "$(label)_k" => () -> discretise!(k_eqn, phi, c))
        push!(fs, "$(label)_p" => () -> discretise!(p_eqn, phi, c))
        push!(fs, "$(label)_U" => () -> discretise!(U_eqn, U, c))
    end
    best = bench_all(fs)
    for (label, _) in asms
        tk = best["$(label)_k"]; tp = best["$(label)_p"]; tu = best["$(label)_U"]
        push!(rows, @sprintf("%-4s %-6s k %8.3f ms   p %8.3f ms   U %8.3f ms   sum %8.3f",
            ix, label, 1000tk, 1000tp, 1000tu, 1000*(2tk+tp+tu)))
    end
    mesh = nothing; GC.gc(true)
end

# short full solve: confirms the whole GPU path still runs and gives a wall time per iteration
runrow = ""
if RUN > 0
    mesh = adapt(backend, load_object(meshpath(IX == "both" ? "i64" : IX)))
    BCs = mkBCs(mesh)
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=nu),
        turbulence=RANS{KOmega}(), energy=Energy{Isothermal}(), domain=mesh)
    mk(sol, rx) = SolverSetup(solver=sol, preconditioner=Jacobi(), convergence=1e-14, relax=rx, rtol=0.1, itmax=1000)
    solvers = (U=mk(Bicgstab(),0.7), p=mk(Cg(),0.3), k=mk(Bicgstab(),0.3), omega=mk(Bicgstab(),0.3))
    schemes = (U=Schemes(time=SteadyState,divergence=LUST,gradient=Gauss),
               p=Schemes(time=SteadyState,gradient=Gauss),
               k=Schemes(time=SteadyState,divergence=Upwind,gradient=Gauss),
               omega=Schemes(time=SteadyState,divergence=Upwind,gradient=Gauss))
    init!() = (initialise!(model.momentum.U, velocity); initialise!(model.momentum.p, 0.0);
        initialise!(model.turbulence.k, k_inlet); initialise!(model.turbulence.omega, omega_inlet);
        initialise!(model.turbulence.nut, nut_inlet))
    rcfg(n, asm) = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=n, write_interval=-1, time_step=1),
        hardware=mkhw(asm), boundaries=BCs)
    io = open(OUT*".runlog","w")
    parts = String[]
    redirect_stdout(io) do; redirect_stderr(io) do; with_logger(SimpleLogger(io)) do
        for (label, asm) in (HAS_FACE ? [("cell",CellAssembly()),("face",FaceAssembly())] : [("cell",nothing)])
            init!(); potential_flow!(model, rcfg(2, asm); ncorrectors=10); run!(model, rcfg(2, asm))
            sync(); GC.gc(true)
            init!(); potential_flow!(model, rcfg(RUN, asm); ncorrectors=10)
            t = @elapsed (res = run!(model, rcfg(RUN, asm)); sync())
            push!(parts, @sprintf("%s %.2f s (%.2f ms/iter) final=%s", label, t, 1000t/RUN,
                string(map(x->round(last(x), sigdigits=4), values(res)))))
        end
    end; end; end
    close(io)
    runrow = join(parts, " | ")
end

open(OUT, "w") do f
    @printf(f, "device=%s index=%s reps=%d face_assembly_available=%s\n", DEV, IX, REPS, HAS_FACE)
    println(f, "sum column = 2k + p + U, i.e. one SIMPLE iteration's assemblies (k and omega same shape)")
    for r in rows; println(f, r); end
    RUN > 0 && println(f, "\nfull solve, $RUN iterations: ", runrow)
end
println("done")
