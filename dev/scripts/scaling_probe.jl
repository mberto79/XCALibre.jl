# Strong-scaling probe for the distributed backward-facing-step case (P1-M3).
# Mesh partitions are cached per rank count so repeated runs measure the solver only.
#   julia --project=<env> dev/scripts/scaling_probe.jl serial <mesh.unv> <iters>
#   julia --project=<env> dev/scripts/scaling_probe.jl drive  <mesh.unv> <iters> <n>...
#   mpiexec -n <n> julia --project=<env> dev/scripts/scaling_probe.jl worker <partdir> <iters>

const MODE = ARGS[1]
const CACHE = joinpath(homedir(), ".cache", "xcal_scaling_probe")

# busiest-core clock, sampled the instant a run ends; sustained load throttles this box badly
function core_mhz()
    mhz = Float64[]
    for line ∈ readlines("/proc/cpuinfo")
        startswith(line, "cpu MHz") && push!(mhz, parse(Float64, split(line, ':')[2]))
    end
    isempty(mhz) ? 0.0 : round(maximum(mhz), digits=1)
end

function bfs_case(domain, iters)
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
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
                        convergence=1e-7, relax=0.2, rtol=0.01, itmax=1000))
    schemes = (U = Schemes(time=SteadyState, divergence=Upwind, gradient=Gauss),
               p = Schemes(time=SteadyState, gradient=Gauss))
    hardware = Hardware(backend=CPU(), workgroup=AutoTune())
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=iters, write_interval=-1, time_step=1),
        hardware=hardware, boundaries=BCs)
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    model, config
end

report(n, ncells, t_short, t_long, short, long, res, mhz) = println(
    "PROBE nranks=$n ncells=$ncells t$short=$(round(t_short, digits=3)) " *
    "t$long=$(round(t_long, digits=3)) " *
    "per_iter=$(round((t_long - t_short) / (long - short), digits=4)) " *
    "mhz=$mhz p=$(res.p[end]) Ux=$(res.Ux[end]) Uy=$(res.Uy[end]) Uz=$(res.Uz[end])")

const SHORT, LONG = 3, parse(Int, ARGS[end])

if MODE == "serial"
    using XCALibre
    activate_multithread(CPU())
    mesh = UNV3D_mesh(ARGS[2], scale=0.001)
    function run_iters(k)
        m, c = bfs_case(mesh, k)
        @elapsed run!(m, c)
    end
    run_iters(1) # absorb compilation before either timed run
    t_short = run_iters(SHORT)
    m, c = bfs_case(mesh, LONG)
    t_long = @elapsed res = run!(m, c)
    mhz = core_mhz()
    report(0, length(mesh.cells), t_short, t_long, SHORT, LONG, res, mhz)

elseif MODE == "worker"
    using XCALibre, PETSc, MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    dm = distribute(ARGS[2]; comm=comm)
    activate_multithread(CPU())
    ncells = MPI.Allreduce(dm.partition.n_owned, +, comm)
    function run_iters(k)
        m, c = bfs_case(dm, k)
        MPI.Barrier(comm)
        t0 = MPI.Wtime()
        res = run!(m, c)
        MPI.Barrier(comm)
        MPI.Wtime() - t0, res
    end
    run_iters(1) # absorb compilation before either timed run
    t_short, _ = run_iters(SHORT)
    t_long, res = run_iters(LONG)
    mhz = core_mhz()
    MPI.Comm_rank(comm) == 0 &&
        report(MPI.Comm_size(comm), ncells, t_short, t_long, SHORT, LONG, res, mhz)

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
    drive(ARGS[2], parse.(Int, ARGS[3:end-1]), ARGS[end])

else
    error("unknown mode $MODE")
end
