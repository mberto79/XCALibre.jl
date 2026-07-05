# Offline partitioning gate: partition_mesh + distribute(dir) must equal online
# distribute(mesh) on every rank (Metis is deterministic on identical input).
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

gmesh = rank == 0 ? bfs_mesh() : nothing
dir = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0)
rank == 0 && partition_mesh(gmesh, nranks; dir)
MPI.Barrier(comm)

dm_off = distribute(dir; comm)
dm_on = distribute(gmesh; comm=comm)

po, pn = dm_off.partition, dm_on.partition
@testset "offline == online (rank $rank)" begin
    @test po.n_owned == pn.n_owned && po.n_ghost == pn.n_ghost
    @test po.local_to_global == pn.local_to_global
    @test po.row_start == pn.row_start && po.row_end == pn.row_end
    @test dm_off.orig_cells == dm_on.orig_cells
    @test dm_off.orig_faces == dm_on.orig_faces
    @test length(dm_off.procs) == length(dm_on.procs)
    @test all(dm_off.procs[i].neighbour == dm_on.procs[i].neighbour &&
              dm_off.procs[i].faces == dm_on.procs[i].faces &&
              dm_off.procs[i].send_cells == dm_on.procs[i].send_cells &&
              dm_off.procs[i].recv_ghosts == dm_on.procs[i].recv_ghosts
              for i ∈ eachindex(dm_off.procs))
    @test dm_off.mesh.cells == dm_on.mesh.cells
    @test dm_off.mesh.faces == dm_on.mesh.faces
    @test dm_off.mesh.cell_neighbours == dm_on.mesh.cell_neighbours
    @test dm_off.mesh.boundary_cellsID == dm_on.mesh.boundary_cellsID
end
rank == 0 && println("OFFLINE == ONLINE n=$nranks")
MPI.Barrier(comm)
rank == 0 && rm(dir; recursive=true)
