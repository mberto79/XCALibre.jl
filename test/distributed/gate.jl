# Fast distributed gate: the subset that catches partitioning, halo, assembly and solver
# regressions at two and three ranks (odd counts give asymmetric halos). The rest of the suite
# is named explicitly to runtests_mpi.jl.
include(joinpath(@__DIR__, "driver.jl"))

mpiexec_available() && run_mpi_tests(
    ["test_halo.jl", "test_partition.jl", "test_assembly.jl",
     "test_laplace.jl", "test_psimple.jl"]; ranks=[2, 3])
