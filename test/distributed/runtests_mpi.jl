# MPI test harness: runs each distributed test file under mpiexec at several rank counts.
# Usage: julia --project test/distributed/runtests_mpi.jl [testfile...] (default: test_halo.jl)
# Rank counts via XCAL_MPI_RANKS (default "1 2 4").
using MPI, Test

files = isempty(ARGS) ? ["test_halo.jl"] : ARGS
ranks = parse.(Int, split(get(ENV, "XCAL_MPI_RANKS", "1 2 4")))
dir = @__DIR__
project = dirname(Base.active_project())
julia = Base.julia_cmd()

# precompile serially first (MPI precompile race); PETSc only if in the environment
run(`$julia --project=$project --startup-file=no -e "using XCALibre, MPI, Test; try using PETSc catch end"`)

@testset "mpi $file n=$n" for file ∈ files, n ∈ ranks
    cmd = `$(MPI.mpiexec()) -n $n $julia --project=$project --startup-file=no $(joinpath(dir, file))`
    out = IOBuffer()
    ok = success(pipeline(cmd; stdout=out, stderr=out))
    ok || print(String(take!(out)))
    @test ok
end
