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

const MESH_ARRAYS = (:cells, :cell_nodes, :cell_faces, :cell_neighbours, :cell_nsign, :faces,
    :face_nodes, :boundaries, :nodes, :node_cells, :boundary_cellsID, :get_float, :get_int)
same_mesh(a, b) = all(getfield(a, k) == getfield(b, k) for k ∈ MESH_ARRAYS)
fields_eq(a, b) = all(getfield(a, k) == getfield(b, k) for k ∈ fieldnames(typeof(a)))
same_part(a, b) = fields_eq(getfield(a, :partition), getfield(b, :partition)) &&
    a.orig_cells == b.orig_cells && a.orig_faces == b.orig_faces &&
    length(a.procs) == length(b.procs) && all(fields_eq(a.procs[i], b.procs[i]) for i ∈ eachindex(a.procs)) &&
    typeof(getfield(a, :mesh)) == typeof(getfield(b, :mesh)) && same_mesh(getfield(a, :mesh), getfield(b, :mesh))

pn = dm_on.partition
@testset "offline == online (rank $rank)" begin
    for k ∈ fieldnames(Partition)
        @test getfield(dm_off.partition, k) == getfield(pn, k)
    end
    for k ∈ MESH_ARRAYS
        @test getfield(dm_off.mesh, k) == getfield(dm_on.mesh, k)
    end
    @test same_part(dm_off, dm_on)
    @test all(getfield(dm_off.mesh, k) isa XCALibre.Mesh.StructArray for k ∈ (:cells, :faces, :nodes))
    @test dm_off.comm == comm
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

# format, kind and rank count are checked at load; the serial kind shares the layout
const D = XCALibre.Distribute
if rank == 0
    bad = mktempdir()
    part0 = joinpath(dir, "rank_0.xdm")
    bytes = read(part0)
    off = length(D._XDM_MAGIC) + 8 * (findfirst(==(:format), D._XDM_KEYS) - 1)
    bytes[off+1:off+8] = reinterpret(UInt8, [Int64(D._XDM_FORMAT - 1)])
    write(joinpath(bad, "rank_0.xdm"), bytes)
    box = UNV3D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "3d_box_1000x1000x1000mm_10.unv"), scale=0.001)
    @testset "part format" begin
        info = mesh_info(part0)
        @test info.kind == :partitioned && info.nranks == nranks && info.rank == 0
        @test info.mesh == Mesh2 && info.TI == XCALibre.Mesh._get_int(gmesh) && info.TF == Float64
        @test info.n_owned == pn.n_owned
        @test D._part_header_ok(part0, nranks)
        @test !D._part_header_ok(part0, nranks + 1)
        @test !D._part_header_ok(joinpath(bad, "rank_0.xdm"), nranks)
        @test !D._parts_match(bad, 1)
        err = try (D._read_part_file(joinpath(bad, "rank_0.xdm"), nranks); nothing) catch e e end
        @test err isa ErrorException && occursin("regenerate with partition_mesh", err.msg)
        err = try (D._read_part_file(part0, nranks + 1); nothing) catch e e end
        @test err isa ErrorException && occursin("mpiexec -n $nranks", err.msg)
        # serial kind: exact round trip, and each loader refuses the other kind naming the right call
        for (name, m) ∈ (("bfs", gmesh), ("box", box))
            path = D._write_mesh_file(joinpath(bad, "$name.xdm"), m)
            @test mesh_info(path).kind == :serial && mesh_info(path).n_ghost == 0
            back = D._read_mesh_file(path)
            @test typeof(back) == typeof(m) && same_mesh(back, m)
            err = try (D._read_part_file(path, 1); nothing) catch e e end
            @test err isa ErrorException && occursin("partition_mesh(mesh, nranks; dir)", err.msg)
        end
        err = try (D._read_mesh_file(part0); nothing) catch e e end
        @test err isa ErrorException && occursin("distribute(dir) under mpiexec -n $nranks", err.msg)
        # a 3D part round trips exactly through the file
        for dm ∈ decompose(box, 3)
            path = D._write_xdm(joinpath(bad, "box_part.xdm"), getfield(dm, :mesh), dm)
            @test same_part(D._read_part_file(path, 3), dm)
        end
    end
    rm(bad; recursive=true)
end

rank == 0 && println("OFFLINE == ONLINE n=$nranks")
MPI.Barrier(comm)
rank == 0 && (rm(dir; recursive=true); rm(dir2; recursive=true))
