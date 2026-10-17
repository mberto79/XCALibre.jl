# One mpiexec session for the two-rank gate (after #205): each file runs in its own module on the
# same ranks, so Julia startup, package load and compilation are paid once instead of per file.
# Files stay runnable standalone (runtests_mpi.jl).
using XCALibre, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# test_mixed_precision: Float32 PETSc corrections for Float64 fields; test_les_wale: nut on owned
# and ghost cells; test_godunov: one smoke case (full set: test_godunov_full.jl); test_failure last:
# every failure path errors on all ranks, and it is the file most likely to hang
const FILES = ["test_partition.jl", "test_halo.jl", "test_assembly.jl", "test_laplace.jl",
    "test_psimple.jl", "test_wall_distance.jl", "test_transpose_stress.jl",
    "test_linear_upwind.jl", "test_reorder.jl", "test_least_squares.jl", "test_forces.jl",
    "test_bound.jl", "test_godunov.jl", "test_unsupported.jl", "test_mixed_precision.jl",
    "test_les_wale.jl", "test_bounding_box.jl", "test_crank_nicolson.jl", "test_postprocess.jl",
    "test_krylov_breakdown.jl", "test_solve_warning.jl", "test_failure.jl"]

timings = Pair{String,Float64}[]
@testset "gate session (rank $rank)" begin
    for file ∈ FILES
        t = @elapsed @testset "$file" begin
            m = Module(Symbol(file))
            Core.eval(m, :(include(p) = Base.include($m, p)))
            Base.include(m, joinpath(@__DIR__, file))
            MPI.Barrier(comm)
        end
        push!(timings, file => t)
    end
end
rank == 0 && foreach(((f, t),) -> println("GATE_TIME ", rpad(f, 32), round(t; digits=1), " s"), timings)
