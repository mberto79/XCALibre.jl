# Per-rank memory breakdown of the distributed BFS case by setup stage; see dev/scripts/INDEX.md.
# `part <mesh.unv> <n> <dir>` writes parts; `worker <dir> <iters> [gc=1] [malloc=1]` measures them.
const MODE = ARGS[1]
const FORCE_GC = "gc=1" in ARGS
const MALLOC = "malloc=1" in ARGS
# `trim=1` calls glibc malloc_trim after each forced collection, so retained-but-free heap is returned
const TRIM = "trim=1" in ARGS
# `gclog=1` makes rank 0 print GC heap stats (bytes_resident) to stderr at each stage
const GCLOG = "gclog=1" in ARGS
# `pre=1` loads a local `M25Pre` package of harvested precompile statements before setup
const PRE = "pre=1" in ARGS
# `gcmax=<MB>` sets the GC memory target at runtime, as `--heap-size-hint` does at start-up
const GCMAX = let i = findfirst(startswith("gcmax="), ARGS); i === nothing ? 0 : parse(Int, ARGS[i][7:end]) end
# `repeat=<k>` adds k full `run!` calls after the staged run, with a forced collection after each
const REPEAT = let i = findfirst(startswith("repeat="), ARGS); i === nothing ? 0 : parse(Int, ARGS[i][8:end]) end
const ARGV = filter(a -> !any(startswith.(a, ("gc=", "malloc=", "repeat=", "trim=", "gclog=", "gcmax=", "pre="))), ARGS)

# kB fields of /proc/self/status, reported in MB
function proc_mb(key)
    for l ∈ eachline("/proc/self/status")
        startswith(l, key) && return parse(Int, split(l)[2]) / 1024
    end
    NaN
end

const SMAPS_KEYS = ("Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty")
# kB fields of /proc/self/smaps_rollup in SMAPS_KEYS order, reported in MB
function smaps_mb()
    d = Dict{String,Float64}()
    for l ∈ eachline("/proc/self/smaps_rollup")
        f = split(l)
        length(f) == 3 && (k = chop(f[1]); k ∈ SMAPS_KEYS) && (d[k] = parse(Int, f[2]) / 1024)
    end
    Tuple(get(d, k, NaN) for k ∈ SMAPS_KEYS)
end

# (private, shared, pss) MB per mapped path; anonymous regions keyed by permissions, so `[anon r-xp]` is JIT code
function smaps_by_path()
    acc = Dict{String,NTuple{3,Float64}}()
    name = ""
    for l ∈ eachline("/proc/self/smaps")
        f = split(l)
        if occursin(r"^[0-9a-f]+-[0-9a-f]+ ", l)
            name = length(f) >= 6 ? join(f[6:end], " ") : "[anon $(f[2])]"
        elseif length(f) == 3 && f[3] == "kB"
            k = chop(f[1]); v = parse(Int, f[2]) / 1024
            i = startswith(k, "Private") ? 1 : startswith(k, "Shared") ? 2 : k == "Pss" ? 3 : 0
            i == 0 && continue
            a = get(acc, name, (0.0, 0.0, 0.0))
            acc[name] = ntuple(j -> a[j] + (j == i) * v, 3)
        end
    end
    sort!(collect(acc); by=x -> -x[2][1])
end

if MODE == "part"
    using XCALibre
    t = @elapsed mesh = UNV3D_mesh(ARGV[2], scale=0.001)
    println("PART load_s=$(round(t, digits=1)) ncells=$(length(mesh.cells)) rss=$(proc_mb("VmRSS")) hwm=$(proc_mb("VmHWM"))")
    t = @elapsed partition_mesh(mesh, parse(Int, ARGV[3]); dir=ARGV[4])
    println("PART partition_s=$(round(t, digits=1)) rss=$(proc_mb("VmRSS")) hwm=$(proc_mb("VmHWM"))")

elseif MODE == "worker"
    using XCALibre, PETSc, MPI, Libdl
    PRE && @eval using M25Pre
    using XCALibre.Solvers: SIMPLE
    using XCALibre.ModelPhysics: initialise
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    const ROWS = Tuple{String,Vararg{Float64,10}}[]
    const MAPS = Pair{String,Vector{Pair{String,NTuple{3,Float64}}}}[]
    # PetscMalloc bytes summed over initialised libraries; zero unless -malloc_debug was set at start-up
    function petsc_malloc_mb()
        mb = 0.0
        for lib ∈ PETSc.petsclibs
            PETSc.initialized(lib) || continue
            m = Ref(0.0)
            ccall(Libdl.dlsym(Libdl.dlopen(lib.petsc_library), :PetscMallocGetCurrentUsage), Cint, (Ref{Float64},), m)
            mb += m[] / 2^20
        end
        mb
    end
    GCLOG && rank == 0 && GC.enable_logging(true)
    GCMAX > 0 && ccall(:jl_gc_set_max_memory, Cvoid, (UInt64,), GCMAX * 2^20)
    function stage(name)
        MPI.Barrier(comm)
        GCLOG && rank == 0 && println(stderr, "STAGE ", name)
        FORCE_GC && GC.gc(true)
        TRIM && ccall(:malloc_trim, Cint, (Csize_t,), 0)
        push!(ROWS, (name, proc_mb("VmRSS"), proc_mb("VmHWM"), Base.gc_live_bytes() / 2^20, MALLOC ? petsc_malloc_mb() : NaN, smaps_mb()...))
        name ∈ ("runtime", "iterations") && push!(MAPS, name => first(smaps_by_path(), 25))
    end
    stage("runtime")
    dm = distribute(ARGV[2]; comm=comm)
    activate_multithread(CPU())
    stage("mesh")
    iters = parse(Int, ARGV[3])
    velocity = [0.5, 0.0, 0.0]
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=dm)
    BCs = assign(region=dm, (
        U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Wall(:wall, [0.0, 0.0, 0.0]),
             Zerogradient(:sides), Zerogradient(:top)],
        p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:wall),
             Extrapolated(:sides), Extrapolated(:top)]))
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.8, rtol=0.1),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-7, relax=0.2, rtol=0.01, itmax=1000))
    schemes = (U = Schemes(time=SteadyState, divergence=Upwind, gradient=Gauss),
               p = Schemes(time=SteadyState, gradient=Gauss))
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=iters, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=AutoTune()), boundaries=BCs)
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    stage("model")
    (; U, p) = model.momentum
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(dm); rDf = FaceScalarField(dm); initialise!(rDf, 1.0)
    nueff = FaceScalarField(dm); divHv = ScalarField(dm)
    stage("aux_fields")
    U_eqn = (Time{schemes.U.time}(U) + Divergence{schemes.U.divergence}(mdotf, U)
        - Laplacian{schemes.U.laplacian}(nueff, U) == -Source(∇p.result)) → VectorEquation(U, BCs.U)
    stage("U_eqn")
    p_eqn = (-Laplacian{schemes.p.laplacian}(rDf, p) == -Source(divHv)) → ScalarEquation(p, BCs.p)
    stage("p_eqn")
    turb, config = initialise(model.turbulence, model, mdotf, p_eqn, config)
    opts = MALLOC ? "-malloc_debug" : ""
    U_w = wrap_eqn(U_eqn, dm, solvers.U, config; petsc_options=opts, label="U")
    p_w = wrap_eqn(p_eqn, dm, solvers.p, config; petsc_options=opts, label="p")
    stage("petsc")
    t = @elapsed res = SIMPLE(model, turb, ∇p, U_w, p_w, config)
    stage("iterations")
    for k ∈ 1:REPEAT
        tk = @elapsed run!(model, config; petsc_options=opts)
        GC.gc(true)
        stage("run!$(k)_$(round(tk, digits=2))s")
    end

    # live bytes by struct, meshes excluded so each field counts only its own arrays
    ex = Union{typeof(dm), typeof(getfield(dm, :mesh))}
    sz(x) = Base.summarysize(x; exclude=ex) / 2^20
    raw(e) = unwrap_eqn(e)
    sizes = [
        "mesh.local" => Base.summarysize(getfield(dm, :mesh)) / 2^20,
        "mesh.partition+procs+orig" => sum(sz(getfield(dm, f)) for f ∈ (:partition, :procs, :orig_cells, :orig_faces)),
        "mesh.halos" => sz(getfield(dm, :halos)),
        "fields.momentum" => sz(model.momentum),
        "fields.momentum+aux" => sz((model.momentum, ∇p, mdotf, rDf, nueff, divHv)),
        "eqns.all" => sz((raw(U_w).equation, raw(p_w).equation)),
        ("U_eqn.$f" => sz(getfield(raw(U_w).equation, f)) for f ∈ fieldnames(typeof(raw(U_w).equation)))...,
        ("p_eqn.$f" => sz(getfield(raw(p_w).equation, f)) for f ∈ fieldnames(typeof(raw(p_w).equation)))...,
    ]
    nown = dm.partition.n_owned; nloc = length(getfield(dm, :mesh).cells)
    out = IOBuffer()
    println(out, "RANK $rank n_owned=$nown n_local=$nloc nfaces=$(length(getfield(dm, :mesh).faces)) iters=$iters gc=$FORCE_GC t_iter_s=$(round(t, digits=2)) p=$(res.p[end]) reshash=$(string(hash(collect(values(res))), base=16))")
    println(out, "stage rss_MB hwm_MB gc_live_MB petsc_malloc_MB ", join(SMAPS_KEYS, "_MB "), "_MB")
    for r ∈ ROWS
        println(out, join((r[1], (round(x, digits=1) for x ∈ r[2:end])...), " "))
    end
    for (st, m) ∈ MAPS, (k, v) ∈ m
        println(out, "map ", st, " priv=", round(v[1], digits=1), " shared=", round(v[2], digits=1), " pss=", round(v[3], digits=1), " ", k)
    end
    for (k, v) ∈ sizes
        println(out, "size ", k, " ", round(v, digits=1))
    end
    s = String(take!(out))
    msgs = MPI.gather(s, comm; root=0)
    rank == 0 && foreach(print, msgs)
end
