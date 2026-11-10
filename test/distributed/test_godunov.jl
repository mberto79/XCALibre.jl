# godunov! distributed vs serial, gate smoke case: laminar HLLC/MUSCL/RK2 with local time
# stepping and the mean cell dt, so the owned-cell reductions of the residuals, the time step
# and `turbulence_dt` all run; forces history included. The full scheme and time-stepping set
# and the 3D box with OpenFOAM output are in test_godunov_full.jl.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "godunov_case.jl"))

gmesh = rank == 0 ? cylinder_mesh() : nothing
dm = distribute(gmesh; comm=comm)
compare_godunov("HLLC MUSCL RK2 LTS mean dt",
    (adaptive=LocalTimeStepping(maxCo=0.5, startCo=0.2, ramp=10, turbulence_dt=:mean),), gmesh, dm, comm)
