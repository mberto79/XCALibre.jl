# One mpiexec session for the whole fast gate: each file runs in its own module on the same ranks,
# so Julia startup and package load are paid once. Files stay runnable standalone (runtests_mpi.jl).
using XCALibre, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# failure last: it leaves every rank in a clean state, but is the file most likely to hang
const FILES = ["test_halo.jl", "test_halo_subcomm.jl", "test_assembly.jl", "test_laplace.jl",
    "test_reorder.jl", "test_transpose_stress.jl", "test_wall_distance.jl", "test_psimple.jl",
    "test_mixed_precision.jl", "test_les_wale.jl", "test_turbulence_sst_wallfn.jl", "test_failure.jl"]

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
rank == 0 && foreach(((f, t),) -> println("GATE_TIME ", rpad(f, 36), round(t; digits=1), " s"), timings)
