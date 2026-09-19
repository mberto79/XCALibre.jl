export repartition

# NEW SECTION: parallel repartition (no rank holds the global mesh)

# implemented by the PETSc extension: 0-based destination rank per owned cell
_parallel_partition(dm, method, petsc_options) = error("repartition needs PETSc: `using PETSc` with a " *
    "PETSc built with ParMETIS or PT-Scotch (conda-forge's petsc has both; PETSc_jll has neither)")

"""
    repartition(dm::DistributedMesh; method=:ptscotch, petsc_options="")

Rebalance a distributed mesh in parallel: PETSc's `MatPartitioning` (`method` `:ptscotch` or
`:parmetis`; PT-Scotch balances cell counts more tightly) partitions the distributed cell graph and cells migrate to their new ranks, so no rank
ever holds the global mesh. Use it after [`distribute`](@ref) of a decomposition that is poorly
balanced, such as `decomposePar -method simple`. Needs `using PETSc` with a PETSc built with the
chosen package. PETSc is initialised here if it is not already, so start-up entries of
`petsc_options` must be given here rather than to the solver.
"""
function repartition(dm::DistributedMesh; method=:ptscotch, petsc_options="")
    method ∈ (:parmetis, :ptscotch) || error("repartition: method must be :ptscotch or :parmetis")
    _migrate(dm, _parallel_partition(dm, method, petsc_options))
end

# each ghost learns its owner's destination through one exchange over the processor patches
function _ghost_values(dm, vals::Vector{Int})
    p = getfield(dm, :partition)
    comm = getfield(dm, :comm)
    out = zeros(Int, p.n_ghost)
    reqs = MPI.Request[]
    bufs = map(getfield(dm, :procs)) do pp
        send, recv = vals[pp.send_cells], Vector{Int}(undef, length(pp.recv_ghosts))
        push!(reqs, MPI.Irecv!(recv, comm; source=pp.neighbour, tag=19))
        push!(reqs, MPI.Isend(send, comm; dest=pp.neighbour, tag=19))
        recv
    end
    MPI.Waitall(reqs)
    for (pp, recv) ∈ zip(getfield(dm, :procs), bufs)
        out[pp.recv_ghosts .- p.n_owned] = recv
    end
    out
end

function _alltoallv(send::Vector{Vector{T}}, comm) where T
    scount = length.(send)
    rcount = MPI.Alltoall(MPI.UBuffer(scount, 1), comm)
    recv = Vector{T}(undef, sum(rcount))
    MPI.Alltoallv!(MPI.VBuffer(reduce(vcat, send; init=T[]), scount), MPI.VBuffer(recv, rcount), comm)
    recv, rcount
end

# per endpoint of a face: old global id, destination, original id, centre, volume
const _EP = 7

function _pack_endpoint!(buf, c, gid, dest, orig, cells)
    push!(buf, gid[c], dest[c], orig[c], cells[c].centre..., cells[c].volume)
end

# cell record: gid, orig, centre, volume, nnodes, node coords, nfaces, face records; a face record is
# patch (0 interior), both endpoints, centre, normal, e, area, delta, weight, orig face, nnodes, coords
function _pack_cells(dm, dest)
    mesh = getfield(dm, :mesh)
    p = getfield(dm, :partition)
    comm = getfield(dm, :comm)
    (; cells, faces, nodes, cell_nodes, cell_faces, face_nodes, boundaries, boundary_cellsID) = mesh
    n = p.n_owned
    alld = vcat(dest, _ghost_values(dm, dest))
    gid = Int.(p.local_to_global)
    orig = Int.(getfield(dm, :orig_cells))
    oface = getfield(dm, :orig_faces)
    patch_of = zeros(Int, length(boundary_cellsID))
    for (b, bd) ∈ enumerate(boundaries), f ∈ bd.IDs_range
        patch_of[f] = b
    end
    bfaces_of = [Int[] for _ ∈ 1:n]
    for (f, c) ∈ enumerate(boundary_cellsID)
        c <= n && push!(bfaces_of[c], f)
    end
    bufs = [Float64[] for _ ∈ 1:MPI.Comm_size(comm)]
    for c ∈ 1:n
        buf = bufs[dest[c]+1]
        push!(buf, gid[c], orig[c], cells[c].centre..., cells[c].volume, length(cells[c].nodes_range))
        foreach(j -> append!(buf, nodes[cell_nodes[j]].coords), cells[c].nodes_range)
        fs = vcat(bfaces_of[c], Int.(view(cell_faces, cells[c].faces_range)))
        push!(buf, length(fs))
        for f ∈ fs
            face = faces[f]
            push!(buf, f <= length(boundary_cellsID) ? patch_of[f] : 0)
            _pack_endpoint!(buf, face.ownerCells[1], gid, alld, orig, cells)
            _pack_endpoint!(buf, face.ownerCells[2], gid, alld, orig, cells)
            push!(buf, face.centre..., face.normal..., face.e..., face.area, face.delta, face.weight,
                oface[f], length(face.nodes_range))
            foreach(j -> append!(buf, nodes[face_nodes[j]].coords), face.nodes_range)
        end
    end
    bufs
end

struct _FaceRec{TF}
    patch::Int
    ep::NTuple{2,NTuple{_EP,Float64}}
    geom::NTuple{12,Float64}
    orig::Int
    nodes::Vector{SVector{3,TF}}
end

function _migrate(dm::DistributedMesh, dest::Vector{Int})
    mesh = getfield(dm, :mesh)
    comm = getfield(dm, :comm)
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    TI, TF = _get_int(mesh), _get_float(mesh)
    FT, CT, NT = eltype(mesh.faces), eltype(mesh.cells), eltype(mesh.nodes)
    length(dest) == getfield(dm, :partition).n_owned && all(d -> 0 <= d < nranks, dest) ||
        error("repartition: one destination rank in 0:$(nranks-1) per owned cell")
    buf, _ = _alltoallv(_pack_cells(dm, dest), comm)

    # unpack: owned cells in original-id order, so a part matches what extraction gives for it
    V(k) = SVector{3,TF}(buf[k], buf[k+1], buf[k+2])
    recs = NamedTuple{(:gid, :orig, :centre, :volume, :nodes, :faces)}[]
    k = 1
    while k <= length(buf)
        gid, orig = Int(buf[k]), Int(buf[k+1])
        centre, volume = V(k + 2), TF(buf[k+5])
        nn = Int(buf[k+6]); k += 7
        cn = [V(k + 3(i - 1)) for i ∈ 1:nn]; k += 3nn
        nf = Int(buf[k]); k += 1
        frs = _FaceRec{TF}[]
        for _ ∈ 1:nf
            patch = Int(buf[k]); k += 1
            ep = (Tuple(buf[k:k+_EP-1]), Tuple(buf[k+_EP:k+2_EP-1])); k += 2_EP
            geom = Tuple(buf[k:k+11]); k += 12
            forig, fnn = Int(buf[k]), Int(buf[k+1]); k += 2
            push!(frs, _FaceRec{TF}(patch, ep, geom, forig, [V(k + 3(i - 1)) for i ∈ 1:fnn])); k += 3fnn
        end
        push!(recs, (; gid, orig, centre, volume, nodes=cn, faces=frs))
    end
    sort!(recs; by=r -> r.orig)
    n_owned = length(recs)
    counts = MPI.Allgather(n_owned, comm)
    row_start = sum(counts[1:rank]; init=0) + 1
    local_of = Dict(r.gid => i for (i, r) ∈ enumerate(recs))

    # ghosts: the owner answers each old global id with its new one
    ghost_info = Dict{Int,NTuple{_EP,Float64}}()
    for r ∈ recs, fr ∈ r.faces, e ∈ fr.ep
        haskey(local_of, Int(e[1])) || (ghost_info[Int(e[1])] = e)
    end
    asks = [Int[] for _ ∈ 1:nranks]
    for (g, e) ∈ ghost_info
        push!(asks[Int(e[2])+1], g)
    end
    foreach(sort!, asks)
    q, qcount = _alltoallv(asks, comm)
    ans = [row_start - 1 + local_of[g] for g ∈ q]
    parts = [ans[s+1:s+c] for (s, c) ∈ zip(cumsum(vcat(0, qcount[1:end-1])), qcount)]
    newg, _ = _alltoallv(parts, comm)
    new_gid = Dict{Int,Int}()
    for (g, ng) ∈ zip(reduce(vcat, asks; init=Int[]), newg)
        new_gid[g] = ng
    end
    ghosts = sort!(collect(keys(ghost_info)); by=g -> (Int(ghost_info[g][2]), new_gid[g]))
    for (i, g) ∈ enumerate(ghosts)
        local_of[g] = n_owned + i
    end
    newid(g) = g ∈ keys(new_gid) ? new_gid[g] : row_start - 1 + local_of[g]

    # a face whose two cells both arrive here is taken from the lower id's record only: the two copies
    # may differ in orientation and last bits when each side built its own geometry
    bf, inf = _FaceRec{TF}[], _FaceRec{TF}[]
    for r ∈ recs, fr ∈ r.faces
        a, b = Int(fr.ep[1][1]), Int(fr.ep[2][1])
        other = a == r.gid ? b : a
        fr.patch == 0 && local_of[other] <= n_owned && other < r.gid && continue
        push!(fr.patch == 0 ? inf : bf, fr)
    end
    sort!(bf; by=fr -> (fr.patch, row_start - 1 + local_of[Int(fr.ep[1][1])]))
    sort!(inf; by=fr -> minmax(newid(Int(fr.ep[1][1])), newid(Int(fr.ep[2][1]))))
    allf = vcat(bf, inf)
    nb = length(bf)

    # nodes: deduplicated by exact coordinates
    node_of = Dict{SVector{3,TF},Int}()
    node_xyz = SVector{3,TF}[]
    nid(x) = get!(() -> (push!(node_xyz, x); length(node_xyz)), node_of, x)
    cell_nodes = TI[]
    nranges = UnitRange{TI}[]
    for r ∈ recs
        s = length(cell_nodes) + 1
        foreach(x -> push!(cell_nodes, nid(x)), r.nodes)
        push!(nranges, UnitRange{TI}(s, length(cell_nodes)))
    end
    face_nodes = TI[]
    new_faces = FT[]
    for fr ∈ allf
        s = length(face_nodes) + 1
        foreach(x -> push!(face_nodes, nid(x)), fr.nodes)
        o1, o2 = local_of[Int(fr.ep[1][1])], local_of[Int(fr.ep[2][1])]
        g = fr.geom
        push!(new_faces, FT(UnitRange{TI}(s, length(face_nodes)), SVector{2,TI}(o1, o2),
            SVector{3,TF}(g[1], g[2], g[3]), SVector{3,TF}(g[4], g[5], g[6]), SVector{3,TF}(g[7], g[8], g[9]),
            TF(g[10]), TF(g[11]), TF(g[12])))
    end

    # cell-face CSR in face order; ghosts keep only their interface faces
    ncl = n_owned + length(ghosts)
    touch = [Tuple{TI,TI,TI}[] for _ ∈ 1:ncl]
    for f ∈ nb+1:length(new_faces)
        o1, o2 = new_faces[f].ownerCells
        push!(touch[o1], (f, o2, 1)); push!(touch[o2], (f, o1, -1))
    end
    cell_faces, cell_neighbours, cell_nsign = TI[], TI[], TI[]
    new_cells = CT[]
    for c ∈ 1:ncl
        s = length(cell_faces) + 1
        for (f, nbc, sg) ∈ touch[c]
            push!(cell_faces, f); push!(cell_neighbours, nbc); push!(cell_nsign, sg)
        end
        fr = UnitRange{TI}(s, length(cell_faces))
        if c <= n_owned
            push!(new_cells, CT(recs[c].centre, recs[c].volume, nranges[c], fr))
        else
            e = ghost_info[ghosts[c-n_owned]]
            push!(new_cells, CT(SVector{3,TF}(e[4], e[5], e[6]), TF(e[7]), UnitRange{TI}(1, 0), fr))
        end
    end

    # node_cells over owned cells, as ranges into one list
    nc = zeros(Int, length(node_xyz))
    foreach(j -> nc[cell_nodes[j]] += 1, eachindex(cell_nodes))
    stops = cumsum(nc)
    starts = stops .- nc .+ 1
    node_cells = zeros(TI, isempty(stops) ? 0 : stops[end])
    fillp = copy(starts)
    for c ∈ 1:n_owned, j ∈ nranges[c]
        node_cells[fillp[cell_nodes[j]]] = c
        fillp[cell_nodes[j]] += 1
    end
    new_nodes = [NT(x, UnitRange{TI}(starts[i], stops[i])) for (i, x) ∈ enumerate(node_xyz)]

    boundaries = map(enumerate(mesh.boundaries)) do (b, bd)
        ids = findall(fr -> fr.patch == b, bf)
        Boundary(bd.name, isempty(ids) ? UnitRange{TI}(nb + 1, nb) : UnitRange{TI}(first(ids), last(ids)))
    end
    boundary_cellsID = TI[new_faces[f].ownerCells[1] for f ∈ 1:nb]
    lmesh = _mesh_like(mesh, new_cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, new_faces,
        face_nodes, boundaries, new_nodes, node_cells, mesh.get_float, mesh.get_int, boundary_cellsID)

    gowner = TI[Int(ghost_info[g][2]) for g ∈ ghosts]
    l2g = vcat(collect(TI, row_start:row_start+n_owned-1), TI[new_gid[g] for g ∈ ghosts])
    partition = Partition(rank, nranks, n_owned, length(ghosts), l2g, vcat(fill(TI(rank), n_owned), gowner),
        row_start, row_start + n_owned - 1)
    procs = map(sort!(unique(gowner))) do q
        pf = TI[f for f ∈ nb+1:length(new_faces) if any(c -> c > n_owned && gowner[c-n_owned] == q, new_faces[f].ownerCells)]
        send = sort!(unique(TI[minimum(new_faces[f].ownerCells) for f ∈ pf]))
        ProcessorPatch(Int(q), pf, send, TI[n_owned + i for i ∈ eachindex(ghosts) if gowner[i] == q])
    end
    orig_cells = vcat(TI[r.orig for r ∈ recs], TI[Int(ghost_info[g][3]) for g ∈ ghosts])
    DistributedMesh(lmesh, partition, procs, orig_cells, TI[fr.orig for fr ∈ allf], HaloCache(), comm)
end
