# Fast distributed gate: the subset that catches partitioning, halo, assembly and solver
# regressions at two and three ranks (odd counts give asymmetric halos). The rest of the suite
# is named explicitly to runtests_mpi.jl.
include(joinpath(@__DIR__, "driver.jl"))

mpiexec_available() && run_mpi_tests(
    ["test_halo.jl", "test_partition.jl", "test_assembly.jl",
     "test_laplace.jl", "test_psimple.jl"]; ranks=[2, 3])

# every failure path errors on all ranks; the rank count does not change which collective is reached
mpiexec_available() && run_mpi_tests(["test_failure.jl"]; ranks=[2])

# no patch of the meshes above empties below six ranks, so the wall-function empty-patch path
# needs its own rank count (10 mm step: rank 2 holds `top` with no faces at n=6)
mpiexec_available() && run_mpi_tests(["test_turbulence_sst_wallfn.jl"]; ranks=[6])
