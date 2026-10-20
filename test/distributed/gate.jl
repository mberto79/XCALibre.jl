# Fast distributed gate (partitioning, halo, assembly, solvers). Two-rank files share one mpiexec
# session (gate_session.jl); rank counts above XCALIBRE_TEST_MAX_RANKS (default 2) are skipped.
# Other files and rank counts: runtests_mpi.jl, e.g. `--ranks=2,3 test_godunov_full.jl`.
include(joinpath(@__DIR__, "driver.jl"))

mpiexec_available() && run_mpi_tests(["gate_session.jl"]; ranks=[2], timeout=2400)

# three ranks: asymmetric halos, a sub-communicator smaller than the world, and the y-band
# partition that empties a wall-function patch on some rank
mpiexec_available() && run_mpi_tests(["gate_session.jl"]; ranks=[3],
    args=["test_halo.jl", "test_halo_subcomm.jl", "test_turbulence_sst_wallfn.jl"])
