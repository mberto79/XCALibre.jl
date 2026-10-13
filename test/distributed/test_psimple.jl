# psimple! gate: BFS + cavity vs serial, per rank under mpiexec (see runtests_mpi.jl).
# Residual histories are sub-tolerance noise under tight inner solves (phase-4 gotcha), so
# only converged fields and ghost consistency are compared.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

function serial_reference(gmesh, bcs, iterations, pref)
    model, config = incompressible_case(gmesh, bcs; iterations)
    simple!(model, config; pref=pref)
    (collect(model.momentum.U.x.values), collect(model.momentum.U.y.values),
     collect(model.momentum.p.values))
end

function run_case(name, build_mesh, bcs, iterations; pref=nothing)
    gmesh = rank == 0 ? build_mesh() : nothing
    ref = rank == 0 ? serial_reference(gmesh, bcs, iterations, pref) : nothing
    Us_x, Us_y, ps = MPI.bcast(ref, comm; root=0)

    dm = distribute(gmesh; comm=comm)
    model, config = incompressible_case(dm, bcs; iterations)
    residuals = run!(model, config; pref=pref)

    dux, duy, dp = field_errors(dm, model, Us_x, Us_y, ps)
    n = dm.partition.n_owned
    nloc = n + dm.partition.n_ghost
    orig = dm.orig_cells
    px = model.momentum.U.x.values

    @testset "$name (rank $rank)" begin
        @test dux < 1e-6
        @test duy < 1e-6
        @test dp < 1e-6
        # ghosts synced to the converged solution
        @test all(abs(px[i] - Us_x[orig[i]]) < 1e-6 for i ∈ n+1:nloc)
        # distributed residuals settle at solver noise floor
        @test maximum(residuals.p[iterations÷2:end]) < 1e-6
        @test maximum(residuals.Ux[iterations÷2:end]) < 1e-6
    end
    rank == 0 && println("PSIMPLE $name n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp")
end

run_case("psimple BFS", bfs_mesh, bfs_bcs, 300)
run_case("psimple cavity", cavity_mesh, cavity_bcs, 300; pref=0.0)

# a continued run! (two calls on the same model) retraces the uninterrupted run on every rank
function continuation_case(build_mesh, bcs, n)
    gmesh = rank == 0 ? build_mesh() : nothing
    dm = distribute(gmesh; comm=comm)
    model1, config1 = incompressible_case(dm, bcs; iterations=2n)
    r1 = run!(model1, config1)
    model2, config2 = incompressible_case(dm, bcs; iterations=n)
    run!(model2, config2)
    r2 = run!(model2, config2)
    nown = dm.partition.n_owned
    rel(a, b) = maximum(abs.(a .- b); init=0.0)/max(maximum(abs.(a); init=0.0), eps())
    owned(model) = (model.momentum.U.x.values[1:nown], model.momentum.p.values[1:nown])
    d = max(maximum(rel.(owned(model1), owned(model2))), rel(r1.p[n+1:2n], r2.p))
    dmax = MPI.Allreduce(d, max, comm)
    @testset "psimple continuation (rank $rank)" begin
        @test dmax <= 1e-10
    end
    rank == 0 && println("PSIMPLE continuation n=$(MPI.Comm_size(comm)) max rel diff = $dmax")
end

continuation_case(bfs_mesh, bfs_bcs, 50)
