# Full distributed suite driver.
#   julia --project test/distributed/runtests_mpi.jl [--ranks=1,2] [testfile...]
include(joinpath(@__DIR__, "driver.jl"))

args = copy(ARGS)
i = findfirst(startswith("--ranks="), args)
ranks = i === nothing ? [1, 2] : parse.(Int, split(split(popat!(args, i), '=')[2], ','))
files = isempty(args) ? ["test_halo.jl"] : args
mpiexec_available() || error("mpiexec is not runnable from $(Base.active_project())")
run_mpi_tests(files; ranks)
