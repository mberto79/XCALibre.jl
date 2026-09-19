# Offline partitioning gate: partition_mesh + distribute(dir), and the rank-uniform
# distribute(reader; dir), must equal online distribute(mesh) on every rank (Metis is
# deterministic on identical input).
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
# the rank-uniform form: every rank runs the same call, only rank 0 reads, and a
# decomposition left by a different rank count must be replaced rather than reused
dir2 = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0)
rank == 0 && partition_mesh(bfs_mesh(), nranks + 1; dir=dir2)
MPI.Barrier(comm)
reads = Ref(0)
dm_uni = distribute(dir=dir2) do
    reads[] += 1
    bfs_mesh()
end
pu = dm_uni.partition
@testset "distribute(reader; dir) (rank $rank)" begin
    @test reads[] == (rank == 0 ? 1 : 0)   # only rank 0 touches the global mesh
    @test pu.nranks == nranks              # the stale (nranks+1)-way parts were replaced
    @test pu.local_to_global == pn.local_to_global
    @test dm_uni.mesh.cells == dm_on.mesh.cells
    @test dm_uni.mesh.faces == dm_on.mesh.faces
end

# a second call with the decomposition already there must not read the mesh again
reads[] = 0
distribute(dir=dir2) do
    reads[] += 1
    bfs_mesh()
end
@testset "decomposition reuse (rank $rank)" begin
    @test reads[] == 0
end

# a part written under another Julia version must fail at load, not inside the solver
if rank == 0
    bad = mktempdir()
    bytes = read(joinpath(dir, "rank_0.jls"))
    nl = findfirst(==(UInt8('\n')), bytes)
    open(joinpath(bad, "rank_0.jls"), "w") do io
        println(io, replace(String(bytes[1:nl-1]), r"julia=\S+" => "julia=0.0.0"))
        write(io, bytes[nl+1:end])
    end
    @testset "part header check" begin
        @test XCALibre.Distribute._part_header_ok(joinpath(dir, "rank_0.jls"), nranks)
        @test !XCALibre.Distribute._part_header_ok(joinpath(bad, "rank_0.jls"), nranks)
        @test !XCALibre.Distribute._parts_match(bad, 1)
        err = try (XCALibre.Distribute._read_part(joinpath(bad, "rank_0.jls"), nranks); nothing) catch e e end
        @test err isa ErrorException && occursin("regenerate with partition_mesh", err.msg)
    end
    rm(bad; recursive=true)
end

rank == 0 && println("OFFLINE == ONLINE n=$nranks")
MPI.Barrier(comm)
rank == 0 && (rm(dir; recursive=true); rm(dir2; recursive=true))
