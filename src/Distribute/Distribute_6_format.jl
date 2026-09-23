export mesh_info

# NEW SECTION: binary mesh format (.xdm)

# layout: magic, header (Int64 per key), mesh block (raw isbits arrays), partition block (empty
# when serial); one code path for both kinds, only the format number is checked for layout (D122)
const _XDM_MAGIC = b"XCALMESH"
const _XDM_FORMAT = 3
const _XDM_BOM = 0x0102030405060708
const _XDM_KEYS = (:bom, :format, :xcalibre_major, :xcalibre_minor, :xcalibre_patch,
    :julia_major, :julia_minor, :julia_patch, :kind, :dim, :TI, :TF,
    :nranks, :rank, :n_owned, :n_ghost,
    :ncells, :ncell_nodes, :ncell_faces, :nfaces, :nface_nodes, :nboundaries, :nboundary_name_bytes,
    :nnodes, :nnode_cells, :nbfaces,
    :row_start, :row_end, :nprocs, :nproc_faces, :nproc_send, :nproc_recv)
const _XDM_KINDS = (:serial, :partitioned)

_xdm_float(bits) = bits == 32 ? Float32 : bits == 64 ? Float64 : error("unsupported float width $bits")
_xdm_int(bits) = bits == 32 ? Int32 : bits == 64 ? Int64 : error("unsupported integer width $bits")

_xdm_types(dim, TI, TF) = (
    cell=Cell{TF,SVector{3,TF},UnitRange{TI}},
    face=(dim == 2 ? Face2D : Face3D){TF,SVector{2,TI},SVector{3,TF},UnitRange{TI}},
    node=Node{SVector{3,TF},UnitRange{TI}},
    mesh=dim == 2 ? Mesh2 : Mesh3)

function _xdm_header(mesh, part)
    TI, TF = _get_int(mesh), _get_float(mesh)
    xv = pkgversion(parentmodule(@__MODULE__))
    names = join(string.(getfield.(mesh.boundaries, :name)), '\n')
    p = part === nothing ? nothing : getfield(part, :partition)
    procs = part === nothing ? ProcessorPatch{Vector{TI}}[] : getfield(part, :procs)
    ncells = length(mesh.cells)
    vals = (_XDM_BOM, _XDM_FORMAT, xv.major, xv.minor, xv.patch, VERSION.major, VERSION.minor, VERSION.patch,
        part === nothing ? 0 : 1, mesh isa Mesh2 ? 2 : 3, 8sizeof(TI), 8sizeof(TF),
        p === nothing ? 1 : p.nranks, p === nothing ? 0 : p.rank,
        p === nothing ? ncells : p.n_owned, p === nothing ? 0 : p.n_ghost,
        ncells, length(mesh.cell_nodes), length(mesh.cell_faces), length(mesh.faces), length(mesh.face_nodes),
        length(mesh.boundaries), sizeof(names), length(mesh.nodes), length(mesh.node_cells),
        length(mesh.boundary_cellsID),
        p === nothing ? 1 : p.row_start, p === nothing ? ncells : p.row_end, length(procs),
        sum(pp -> length(pp.faces), procs; init=0), sum(pp -> length(pp.send_cells), procs; init=0),
        sum(pp -> length(pp.recv_ghosts), procs; init=0))
    NamedTuple{_XDM_KEYS}(Int64.(vals)), names
end

function _xdm_write_array(io, v, ::Type{T}) where T
    isbitstype(T) && eltype(v) === T || error("cannot write $(eltype(v)) as $T")
    write(io, convert(Vector{T}, v)) # packs a StructArray into records; a no-op on a Vector
end

function _xdm_read_array(io, ::Type{T}, n) where T
    v = Vector{T}(undef, n)
    read!(io, v)
end

# mesh arrays in header order, then the partition block when `part` is a DistributedMesh
function _write_xdm(path, mesh, part=nothing)
    h, names = _xdm_header(mesh, part)
    TI = _get_int(mesh)
    T = _xdm_types(h.dim, TI, _get_float(mesh))
    open(path, "w") do io
        write(io, _XDM_MAGIC)
        write(io, collect(values(h)))
        _xdm_write_array(io, mesh.cells, T.cell)
        foreach(v -> _xdm_write_array(io, v, TI),
            (mesh.cell_nodes, mesh.cell_faces, mesh.cell_neighbours, mesh.cell_nsign))
        _xdm_write_array(io, mesh.faces, T.face)
        _xdm_write_array(io, mesh.face_nodes, TI)
        write(io, names)
        _xdm_write_array(io, UnitRange{TI}[b.IDs_range for b ∈ mesh.boundaries], UnitRange{TI})
        _xdm_write_array(io, mesh.nodes, T.node)
        _xdm_write_array(io, mesh.node_cells, TI)
        _xdm_write_array(io, mesh.boundary_cellsID, TI)
        _xdm_write_array(io, [SVector{3,_get_float(mesh)}(mesh.get_float)], SVector{3,_get_float(mesh)})
        _xdm_write_array(io, [mesh.get_int], UnitRange{TI})
        part === nothing && return
        p = getfield(part, :partition)
        procs = getfield(part, :procs)
        foreach(v -> _xdm_write_array(io, v, TI), (p.local_to_global, p.owner,
            getfield(part, :orig_cells), getfield(part, :orig_faces)))
        _xdm_write_array(io, TI[pp.neighbour for pp ∈ procs], TI)
        for f ∈ (:faces, :send_cells, :recv_ghosts)
            _xdm_write_array(io, TI[length(getfield(pp, f)) for pp ∈ procs], TI)
            foreach(pp -> _xdm_write_array(io, getfield(pp, f), TI), procs)
        end
    end
    path
end

function _read_xdm_header(io, path)
    magic = read(io, length(_XDM_MAGIC))
    magic == _XDM_MAGIC || error("$path is not an XCALibre mesh file; regenerate with partition_mesh")
    h = NamedTuple{_XDM_KEYS}(Tuple(_xdm_read_array(io, Int64, length(_XDM_KEYS))))
    h.bom == _XDM_BOM || error("$path was written on a machine of the other byte order")
    h.format == _XDM_FORMAT || error("$path was written with format=$(h.format) but this XCALibre reads " *
        "format=$(_XDM_FORMAT); regenerate with partition_mesh")
    h
end

"""
    mesh_info(path)

Header of a binary mesh file written by [`partition_mesh`](@ref), as a `NamedTuple`: `kind`
(`:serial` or `:partitioned`), `nranks` and `rank` (1 and 0 for a serial mesh), `mesh` (`Mesh2` or
`Mesh3`), the integer and float types `TI` and `TF`, `n_owned` and `n_ghost` cells, the format number
and the XCALibre and Julia versions that wrote it, and the length of every stored array.
"""
function mesh_info(path)
    h = open(io -> _read_xdm_header(io, path), path)
    (; format=h.format, xcalibre=VersionNumber(h.xcalibre_major, h.xcalibre_minor, h.xcalibre_patch),
        julia=VersionNumber(h.julia_major, h.julia_minor, h.julia_patch), kind=_XDM_KINDS[h.kind+1],
        mesh=h.dim == 2 ? Mesh2 : Mesh3, TI=_xdm_int(h.TI), TF=_xdm_float(h.TF),
        (k => getfield(h, k) for k ∈ _XDM_KEYS[13:end])...)
end

function _read_xdm_body(io, h)
    TI, TF = _xdm_int(h.TI), _xdm_float(h.TF)
    T = _xdm_types(h.dim, TI, TF)
    cells = _xdm_read_array(io, T.cell, h.ncells)
    cell_nodes = _xdm_read_array(io, TI, h.ncell_nodes)
    cell_faces, cell_neighbours, cell_nsign = (_xdm_read_array(io, TI, h.ncell_faces) for _ ∈ 1:3)
    faces = _xdm_read_array(io, T.face, h.nfaces)
    face_nodes = _xdm_read_array(io, TI, h.nface_nodes)
    names = h.nboundaries == 0 ? String[] : split(String(read(io, h.nboundary_name_bytes)), '\n')
    ranges = _xdm_read_array(io, UnitRange{TI}, h.nboundaries)
    boundaries = [Boundary(Symbol(n), r) for (n, r) ∈ zip(names, ranges)]
    nodes = _xdm_read_array(io, T.node, h.nnodes)
    node_cells = _xdm_read_array(io, TI, h.nnode_cells)
    boundary_cellsID = _xdm_read_array(io, TI, h.nbfaces)
    get_float = _xdm_read_array(io, SVector{3,TF}, 1)[1]
    get_int = _xdm_read_array(io, UnitRange{TI}, 1)[1]
    mesh = T.mesh(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
        boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID)
    h.kind == 0 && return mesh
    nlocal = h.n_owned + h.n_ghost
    l2g, owner, orig_cells = (_xdm_read_array(io, TI, nlocal) for _ ∈ 1:3)
    orig_faces = _xdm_read_array(io, TI, h.nfaces)
    neighbours = _xdm_read_array(io, TI, h.nprocs)
    lists = map((h.nproc_faces, h.nproc_send, h.nproc_recv)) do total
        lens = _xdm_read_array(io, TI, h.nprocs)
        sum(lens; init=0) == total || error("corrupt partition block: patch lengths do not sum to $total")
        [_xdm_read_array(io, TI, n) for n ∈ lens]
    end
    procs = [ProcessorPatch(Int(neighbours[i]), lists[1][i], lists[2][i], lists[3][i]) for i ∈ 1:h.nprocs]
    partition = Partition(h.rank, h.nranks, h.n_owned, h.n_ghost, l2g, owner, h.row_start, h.row_end)
    DistributedMesh(mesh, partition, procs, orig_cells, orig_faces, HaloCache(), MPI.COMM_NULL)
end

function _read_xdm(path, check)
    open(path) do io
        h = _read_xdm_header(io, path)
        check(h)
        out = _read_xdm_body(io, h)
        eof(io) || error("$path has trailing bytes after its declared arrays; regenerate with partition_mesh")
        out
    end
end

_check_part(path, nranks) = h -> begin
    h.kind == 1 || error("$path holds a serial mesh, not a partition; decompose it with " *
        "partition_mesh(mesh, nranks; dir) and load the parts with distribute(dir) under mpiexec -n <nranks>")
    h.nranks == nranks || error("partition $path was written for nranks=$(h.nranks) but this run has " *
        "nranks=$nranks; run distribute(dir) under mpiexec -n $(h.nranks) or regenerate with partition_mesh")
end

_read_part_file(path, nranks) = _read_xdm(path, _check_part(path, nranks))

# serial kind: built here for the shared layout, published with the serial mesh format off main (D120)
_write_mesh_file(path, mesh) = _write_xdm(path, mesh)
_read_mesh_file(path) = _read_xdm(path, h -> h.kind == 0 || error("$path is part $(h.rank) of a " *
    "$(h.nranks)-rank decomposition; load it with distribute(dir) under mpiexec -n $(h.nranks)"))

_part_header_ok(path, nranks) = try
    open(io -> (_check_part(path, nranks)(_read_xdm_header(io, path)); true), path)
catch
    false
end
