# Fast distributed gate: one three-rank mpiexec session (gate_session.jl) runs the files that catch
# partitioning, halo, assembly and solver regressions; three ranks give asymmetric halos and the
# sst wall-function test its empty patches. The two-rank and wider sweeps are named explicitly to
# runtests_mpi.jl (nightly/manual).
using Test
using XCALibre
include(joinpath(@__DIR__, "driver.jl"))

# partitioning needs no MPI: serial, in-process
let m = Module(:test_partition)
    Base.include(m, joinpath(@__DIR__, "test_partition.jl"))
end

mpiexec_available() && run_mpi_tests(["gate_session.jl"]; ranks=[3])
