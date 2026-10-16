# Fast distributed gate: the subset that catches partitioning, halo, assembly and solver
# regressions. The two-rank files share one mpiexec session (gate_session.jl); each launch costs
# minutes of compilation. Rank counts above XCALIBRE_TEST_MAX_RANKS (default 2) are skipped, see
# driver.jl. The rest of the suite, and other rank counts, are named explicitly to
# runtests_mpi.jl, e.g. `--ranks=2,3 test_godunov_full.jl`.
include(joinpath(@__DIR__, "driver.jl"))

mpiexec_available() && run_mpi_tests(["gate_session.jl"]; ranks=[2], timeout=2400)

# odd rank count: asymmetric halos
mpiexec_available() && run_mpi_tests(["test_halo.jl"]; ranks=[3])

# no patch of the meshes above empties below six ranks, so the wall-function empty-patch path
# needs its own rank count (10 mm step: rank 2 holds `top` with no faces at n=6)
mpiexec_available() && run_mpi_tests(["test_turbulence_sst_wallfn.jl"]; ranks=[6])
