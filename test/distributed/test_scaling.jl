# Scaling gate: fixed-work distributed Laplace solve on a larger 2D mesh across rank counts.
# Driver mode (default): spawns itself under mpiexec for each rank count (ARGS, default 1 2 4 8),
# prints a speedup table and asserts speedup at the largest count (XCAL_MIN_SPEEDUP, default 1.2).
# Test at top level: macros in the else branch are lowered before a conditional `using` runs
using MPI, Test

if get(ENV, "XCAL_SCALING_WORKER", "") == "1"
    using XCALibre, PETSc
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    include(joinpath(@__DIR__, "laplace_case.jl"))

    gmesh = rank == 0 ? fine2d_mesh() : nothing
    ncells = MPI.bcast(rank == 0 ? length(gmesh.cells) : 0, comm; root=0)
    dm = distribute(gmesh; comm=comm)
    model, config = laplace_case(dm, fine2d_bcs)
    deqn = build_deqn(dm, model, config)
    T = model.energy.T

    solve_iter!() = solve_equation!(deqn, T, config.boundaries.T, config.solvers, config; time=1.0)
    reset!() = initialise!(T, 15.0)

    reset!(); solve_iter!(); reset!(); solve_iter!() # warmup
    reps = 10
    ts = zeros(reps)
    for r ∈ 1:reps
        reset!()
        MPI.Barrier(comm)
        t0 = MPI.Wtime()
        solve_iter!()
        MPI.Barrier(comm)
        ts[r] = MPI.Wtime() - t0
    end
    t = sort(ts)[reps ÷ 2 + 1]
    rank == 0 && println("SCALING nranks=$(MPI.Comm_size(comm)) ncells=$ncells t_solve=$t")
else
    ranks = isempty(ARGS) ? [1, 2, 4, 8] : parse.(Int, ARGS)
    julia = Base.julia_cmd()
    project = dirname(Base.active_project())
    # precompile serially first (MPI precompile race)
    run(`$julia --project=$project --startup-file=no -e "using XCALibre, MPI, PETSc"`)
    times = Float64[]
    for n ∈ ranks
        cmd = addenv(`$(MPI.mpiexec()) -n $n $julia --project=$project --startup-file=no $(@__FILE__)`,
            "XCAL_SCALING_WORKER" => "1")
        out = read(cmd, String)
        m = match(r"t_solve=([0-9.eE+-]+)", out)
        m === nothing && (print(out); error("scaling worker n=$n produced no SCALING line"))
        push!(times, parse(Float64, m[1]))
    end
    println("SCALING table (median fixed-work solve)")
    for (n, t) ∈ zip(ranks, times)
        println("  n=$n  t=$(round(t*1000, digits=2)) ms  speedup=$(round(times[1]/t, digits=2))")
    end
    min_speedup = parse(Float64, get(ENV, "XCAL_MIN_SPEEDUP", "1.2"))
    @testset "scaling" begin
        @test times[1] / times[end] > min_speedup
    end
end
