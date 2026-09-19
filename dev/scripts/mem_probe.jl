# Per-rank memory breakdown of the distributed BFS case by setup stage; see dev/scripts/INDEX.md.
# `part <mesh.unv> <n> <dir>` writes parts; `worker <dir> <iters> [gc=1] [malloc=1]` measures them.
const MODE = ARGS[1]
const FORCE_GC = "gc=1" in ARGS
const MALLOC = "malloc=1" in ARGS
# `repeat=<k>` adds k full `run!` calls after the staged run, with a forced collection after each
const REPEAT = let i = findfirst(startswith("repeat="), ARGS); i === nothing ? 0 : parse(Int, ARGS[i][8:end]) end
const ARGV = filter(a -> !any(startswith.(a, ("gc=", "malloc=", "repeat="))), ARGS)

# kB fields of /proc/self/status, reported in MB
function proc_mb(key)
    for l ∈ eachline("/proc/self/status")
        startswith(l, key) && return parse(Int, split(l)[2]) / 1024
    end
    NaN
end

if MODE == "part"
    using XCALibre
    t = @elapsed mesh = UNV3D_mesh(ARGV[2], scale=0.001)
    println("PART load_s=$(round(t, digits=1)) ncells=$(length(mesh.cells)) rss=$(proc_mb("VmRSS")) hwm=$(proc_mb("VmHWM"))")
    t = @elapsed partition_mesh(mesh, parse(Int, ARGV[3]); dir=ARGV[4])
    println("PART partition_s=$(round(t, digits=1)) rss=$(proc_mb("VmRSS")) hwm=$(proc_mb("VmHWM"))")

elseif MODE == "worker"
    using XCALibre, PETSc, MPI, Libdl
    using XCALibre.Solvers: SIMPLE
    using XCALibre.ModelPhysics: initialise
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    const ROWS = Tuple{String,Float64,Float64,Float64,Float64}[]
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
    function stage(name)
        MPI.Barrier(comm)
        FORCE_GC && GC.gc(true)
        push!(ROWS, (name, proc_mb("VmRSS"), proc_mb("VmHWM"), Base.gc_live_bytes() / 2^20, MALLOC ? petsc_malloc_mb() : NaN))
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
        run!(model, config; petsc_options=opts)
        GC.gc(true)
        stage("run!$k")
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
    println(out, "stage rss_MB hwm_MB gc_live_MB petsc_malloc_MB")
    for r ∈ ROWS
        println(out, join((r[1], (round(x, digits=1) for x ∈ r[2:end])...), " "))
    end
    for (k, v) ∈ sizes
        println(out, "size ", k, " ", round(v, digits=1))
    end
    s = String(take!(out))
    msgs = MPI.gather(s, comm; root=0)
    rank == 0 && foreach(print, msgs)
end
