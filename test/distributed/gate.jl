# Fast distributed gate: the subset that catches partitioning, halo, assembly and solver
# regressions, one rank count per file (each mpiexec launch costs minutes on the coverage CI
# job). The rest of the suite, and other rank counts, are named explicitly to runtests_mpi.jl,
# e.g. `--ranks=2,3 test_godunov_full.jl`.
include(joinpath(@__DIR__, "driver.jl"))

# two ranks; test_mixed_precision: PETSc_jll's Float32 library solves corrections for Float64
# fields; test_les_wale: algebraic nut from gradU matches serial on owned and ghost cells;
# test_failure: every failure path errors on all ranks (the rank count does not change which
# collective is reached); test_godunov: one density-based smoke case (the full set is
# test_godunov_full.jl)
mpiexec_available() && run_mpi_tests(
    ["test_partition.jl", "test_assembly.jl", "test_laplace.jl", "test_psimple.jl",
     "test_wall_distance.jl", "test_transpose_stress.jl", "test_linear_upwind.jl",
     "test_reorder.jl", "test_least_squares.jl", "test_forces.jl", "test_bound.jl",
     "test_godunov.jl", "test_unsupported.jl", "test_mixed_precision.jl", "test_les_wale.jl",
     "test_postprocess.jl", "test_failure.jl"]; ranks=[2])

# odd rank count: asymmetric halos
mpiexec_available() && run_mpi_tests(["test_halo.jl"]; ranks=[3])

# no patch of the meshes above empties below six ranks, so the wall-function empty-patch path
# needs its own rank count (10 mm step: rank 2 holds `top` with no faces at n=6)
mpiexec_available() && run_mpi_tests(["test_turbulence_sst_wallfn.jl"]; ranks=[6])
