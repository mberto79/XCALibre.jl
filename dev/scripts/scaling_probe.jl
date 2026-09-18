# Strong-scaling probe for the distributed backward-facing-step case; see dev/scripts/INDEX.md.
# Partitions are cached per rank count so repeated runs measure the solver only.
const MODE = ARGS[1]
# optional `pc=<jacobi|boomeramg|gamg|ic0|ilu0>` may appear anywhere; positional args ignore it
const PCNAME = let i = findfirst(a -> startswith(a, "pc="), ARGS)
    i === nothing ? "jacobi" : ARGS[i][4:end]
end
const PCREUSE = let i = findfirst(a -> startswith(a, "reuse="), ARGS)
    i === nothing ? nothing : parse(Int, ARGS[i][7:end])
end
# optional `dev=cuda` runs the worker on the GPU (needs CUDA and a CUDA-enabled PETSc in the env)
const DEV = let i = findfirst(a -> startswith(a, "dev="), ARGS)
    i === nothing ? "cpu" : ARGS[i][5:end]
end
if DEV == "cuda" && MODE == "worker"
    using XCALibre, PETSc, MPI, CUDA
end
const ARGV = filter(a -> !any(startswith.(a, ("pc=", "reuse=", "dev=", "wait="))), ARGS)
const CACHE = joinpath(homedir(), ".cache", "xcal_scaling_probe")

# busiest-core clock, sampled the instant a run ends; sustained load throttles this box badly
function core_mhz()
    mhz = Float64[]
    for line ∈ readlines("/proc/cpuinfo")
        startswith(line, "cpu MHz") && push!(mhz, parse(Float64, split(line, ':')[2]))
    end
    isempty(mhz) ? 0.0 : round(maximum(mhz), digits=1)
end

function bfs_case(domain, iters; petsc_options="", pc=PCNAME, backend=CPU())
    velocity = [0.5, 0.0, 0.0]
    nu = 1e-3
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu = nu),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = domain)
    BCs = assign(region=domain, (
        U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Wall(:wall, [0.0, 0.0, 0.0]),
             Zerogradient(:sides), Zerogradient(:top)],
        p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:wall),
             Extrapolated(:sides), Extrapolated(:top)]))
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                        convergence=1e-7, relax=0.8, rtol=0.1),
        p = SolverSetup(solver=Cg(),
                        preconditioner = pc == "boomeramg" ?
                                           (PCREUSE === nothing ? BoomerAMG() : BoomerAMG(freeze=PCREUSE)) :
                                         pc == "gamg" ?
                                           (PCREUSE === nothing ? GAMG() : GAMG(freeze=PCREUSE)) :
                                         pc == "ic0" ? IC0GPU() : pc == "ilu0" ? ILU0GPU() : Jacobi(),
                        convergence=1e-7, relax=0.2, rtol=0.01, itmax=1000))
    schemes = (U = Schemes(time=SteadyState, divergence=Upwind, gradient=Gauss),
               p = Schemes(time=SteadyState, gradient=Gauss))
    hardware = Hardware(backend=backend, workgroup=backend isa CPU ? AutoTune() : 32)
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=iters, write_interval=-1, time_step=1),
        hardware=hardware, boundaries=BCs)
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    model, config, petsc_options
end

report(n, ncells, t_short, t_long, short, long, res, mhz) = println(
    "PROBE dev=$DEV pc=$PCNAME reuse=$PCREUSE nranks=$n ncells=$ncells t$short=$(round(t_short, digits=3)) " *
    "t$long=$(round(t_long, digits=3)) " *
    "per_iter=$(round((t_long - t_short) / (long - short), digits=4)) " *
    "mhz=$mhz p=$(res.p[end]) Ux=$(res.Ux[end]) Uy=$(res.Uy[end]) Uz=$(res.Uz[end])")

const SHORT, LONG = 3, parse(Int, ARGV[end])

if MODE == "serial"
    using XCALibre
    activate_multithread(CPU())
    mesh = UNV3D_mesh(ARGV[2], scale=0.001)
    function run_iters(k)
        m, c, _ = bfs_case(mesh, k)
        @elapsed run!(m, c)
    end
    run_iters(1) # absorb compilation before either timed run
    t_short = run_iters(SHORT)
    m, c, _ = bfs_case(mesh, LONG)
    t_long = @elapsed res = run!(m, c)
    mhz = core_mhz()
    report(0, length(mesh.cells), t_short, t_long, SHORT, LONG, res, mhz)

elseif MODE == "worker"
    using XCALibre, PETSc, MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    backend = DEV == "cuda" ? CUDABackend() : CPU()
    DEV == "cuda" && bind_device!(backend, MPI.Comm_rank(comm))
    dm = distribute(ARGV[2]; comm=comm)
    DEV == "cuda" && (dm = adapt(backend, dm))
    activate_multithread(CPU())
    ncells = MPI.Allreduce(dm.partition.n_owned, +, comm)
    opts = length(ARGV) >= 4 ? ARGV[3] : ""
    function run_iters(k)
        m, c, o = bfs_case(dm, k; petsc_options=opts, backend)
        MPI.Barrier(comm)
        t0 = MPI.Wtime()
        res = run!(m, c; petsc_options=o)
        MPI.Barrier(comm)
        MPI.Wtime() - t0, res
    end
    run_iters(1) # absorb compilation before either timed run
    # optional `wait=1`: time each rank's barrier wait on entry to every PETSc assembly
    WAIT = Ref(0.0)
    if "wait=1" in ARGS
        ext = Base.get_extension(XCALibre, :XCALibrePETScExt)
        # copy of the ext's passemble! body with a timed barrier in front; keep in step with the ext
        @eval ext function passemble!(s::XPETScSolver, eqn, partition; component=nothing)
            t0 = time(); MPI.Barrier(MPI.COMM_WORLD); $WAIT[] += time() - t0
            _set_values!(s.petsclib, s.A, _nzval(_A(eqn)), s.sync)
            PETSc.withlocalarray!(s.b; read=false, write=true) do arr
                copyto!(arr, view(_b(eqn, component), 1:s.n_owned))
            end
            s
        end
    end
    t_short, _ = run_iters(SHORT)
    t_long, res = run_iters(LONG)
    mhz = core_mhz()
    MPI.Comm_rank(comm) == 0 &&
        report(MPI.Comm_size(comm), ncells, t_short, t_long, SHORT, LONG, res, mhz)
    if "wait=1" in ARGS
        w = MPI.Gather(WAIT[], comm; root=0)
        MPI.Comm_rank(comm) == 0 && println("WAIT per-rank barrier wait s (short+long runs): ",
            join(round.(w, digits=3), " "))
    end

elseif MODE == "drive"
    using XCALibre, MPI
    function drive(mesh_path, counts, iters)
        tag = first(splitext(basename(mesh_path)))
        dirs = [joinpath(CACHE, "$(tag)_n$n") for n ∈ counts]
        if any(!isdir, dirs)
            mesh = UNV3D_mesh(mesh_path, scale=0.001)
            for (n, dir) ∈ zip(counts, dirs)
                isdir(dir) || partition_mesh(mesh, n; dir=dir)
            end
        end
        GC.gc(true) # the global mesh must be gone before the workers claim memory
        julia = Base.julia_cmd()
        project = dirname(Base.active_project())
        for (n, dir) ∈ zip(counts, dirs)
            run(`$(MPI.mpiexec()) -n $n --bind-to core --map-by core $julia --project=$project --startup-file=no $(@__FILE__) worker $dir $iters`)
        end
    end
    drive(ARGV[2], parse.(Int, ARGV[3:end-1]), ARGV[end])

else
    error("unknown mode $MODE")
end
