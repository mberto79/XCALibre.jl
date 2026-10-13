# psimple! gate: BFS + cavity vs serial, per rank under mpiexec (see runtests_mpi.jl).
# Fields and ghost consistency are compared at the end; residual histories only over the first
# iterations, before they fall to sub-tolerance noise (tight inner solves, phase-4 gotcha).
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

const NHIST = 6

function serial_reference(gmesh, bcs, iterations, pref, consistent)
    model, config = incompressible_case(gmesh, bcs; iterations)
    res = simple!(model, config; pref=pref, consistent=consistent)
    (collect(model.momentum.U.x.values), collect(model.momentum.U.y.values),
     collect(model.momentum.p.values), res.p[1:NHIST], res.Ux[1:NHIST])
end

function run_case(name, build_mesh, bcs, iterations; pref=nothing, consistent=false)
    gmesh = rank == 0 ? build_mesh() : nothing
    ref = rank == 0 ? serial_reference(gmesh, bcs, iterations, pref, consistent) : nothing
    Us_x, Us_y, ps, hist_p, hist_ux = MPI.bcast(ref, comm; root=0)

    dm = distribute(gmesh; comm=comm)
    model, config = incompressible_case(dm, bcs; iterations)
    residuals = run!(model, config; pref=pref, consistent=consistent)

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
        # early residual history retraces the serial run
        @test residuals.p[1:NHIST] ≈ hist_p rtol = 1e-5
        @test residuals.Ux[1:NHIST] ≈ hist_ux rtol = 1e-5
    end
    rank == 0 && println("PSIMPLE $name n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp")
end

run_case("psimple BFS", bfs_mesh, bfs_bcs, 70)
run_case("psimple cavity", cavity_mesh, cavity_bcs, 70; pref=0.0)
# consistent coupling: rDt is exchanged before it is interpolated to the faces
run_case("psimple BFS consistent", bfs_mesh, bfs_bcs, 70; consistent=true)

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
    dfield = MPI.Allreduce(maximum(rel.(owned(model1), owned(model2))), max, comm)
    dres = MPI.Allreduce(rel(r1.p[n+1:2n], r2.p), max, comm)
    # threaded atomic boundary sums vary in order, so identical runs differ by ~1e-14 in the fields
    # but up to ~1.1e-7 in the p residual, which sits near atol (1e-10..7e-10; 2-3 ranks, 4-8
    # threads): fields stay strict, residuals get 100x that spread
    @testset "psimple continuation (rank $rank)" begin
        @test dfield <= 1e-10
        @test dres <= 1e-5
    end
    rank == 0 && println("PSIMPLE continuation n=$(MPI.Comm_size(comm)) rel diff fields = $dfield residuals = $dres")
end

continuation_case(bfs_mesh, bfs_bcs, 10)
