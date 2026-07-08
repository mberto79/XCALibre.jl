# MPI scatter smoke test: mpiexec -n N julia --project dev/mpi_smoke.jl
using XCALibre, MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
mesh = rank == 0 ?
    UNV3D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "3d_box_1000x1000x1000mm_10.unv"), scale=0.001) :
    nothing
dm = distribute(mesh; comm=comm)
tot = MPI.Allreduce(dm.partition.n_owned, +, comm)
println("rank $rank: $dm")
rank == 0 && println("SMOKE $(tot == 1000 ? "OK" : "FAIL"): total owned = $tot, ranks = $(MPI.Comm_size(comm))")
