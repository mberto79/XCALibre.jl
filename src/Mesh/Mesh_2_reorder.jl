export reorder_mesh!

"""
    reorder_mesh!(mesh::Union{Mesh2,Mesh3}; method=:rcm) -> mesh

Reorder the cells, faces and nodes of a host `mesh` in place for memory locality, so the neighbours
that cell-based kernels read sit close in memory. Geometry, face orientation and boundary patches
are unchanged; only the ordering differs. Arrays are permuted in place: the extra memory is a few
index vectors of the mesh's integer type and one buffer the size of the largest connectivity list.

- `method=:rcm`: reverse Cuthill-McKee ordering of the cell graph (bounded index distance between
  neighbours).
- `method=:morton`: Z-order curve of the cell centres (compact groups of consecutive cells).

Internal faces follow their lowest-numbered cell, boundary faces their owner cell within each patch,
and nodes the order in which the cells first reach them. Fields built on the reordered mesh are in
the new cell order; `output=OpenFOAM()` rewrites `constant/polyMesh` in that order when the files
there are numbered differently. Meshes from octree or block-structured generators are often ordered
already; unstructured (e.g. tetrahedral) meshes usually gain most. A mesh is left unchanged unless
the new order shortens the mean index distance between neighbouring cells by at least 10%, so an
ordered mesh is read back as it is. Serial meshes only; call before `adapt`.
"""
reorder_mesh!(mesh::Union{Mesh2,Mesh3}; method::Symbol=:rcm) = (_reorder_mesh!(mesh, method); mesh)

# reorders in place; returns the cell permutation (perm[new] = old), or nothing when the mesh is
# already ordered
function _reorder_mesh!(mesh, method)
    TI = _get_int(mesh)
    ncells = length(mesh.cell_volume)
    nfaces = length(mesh.face_area)
    nnodes = length(mesh.node_coords)
    nbfaces = length(mesh.boundary_cellsID)
    (; cell_faces_range, cell_neighbours, face_ownerCells) = mesh

    perm = method === :rcm ? _rcm(ncells, u -> cell_faces_range[u], cell_neighbours, TI) :
        method === :morton ? _morton_permutation(mesh.cell_centre, TI) :
        throw(ArgumentError("method must be :rcm or :morton"))
    iperm = _invperm(perm)
    _shortens_neighbour_distance(face_ownerCells, nbfaces, iperm) || return nothing

    # boundary faces by owner within each patch, internal faces by their lower cell
    fperm = collect(TI(1):TI(nfaces))
    for b ∈ mesh.boundaries
        sort!(view(fperm, b.IDs_range), by = f -> iperm[face_ownerCells[f][1]])
    end
    sort!(view(fperm, (nbfaces + 1):nfaces), by = f -> _face_key(iperm, face_ownerCells[f]))
    ifperm = _invperm(fperm)

    # nodes in first-touch order of the new cells
    done = falses(max(ncells, nfaces, nnodes))
    nperm = similar(perm, nnodes)
    k = 0
    for c ∈ perm, n ∈ view(mesh.cell_nodes, mesh.cell_nodes_range[c])
        done[n] || (done[n] = true; nperm[k += 1] = n)
    end
    for n ∈ 1:nnodes
        done[n] || (nperm[k += 1] = n)
    end
    inperm = _invperm(nperm)

    buffer = similar(mesh.cell_nodes, max(length(mesh.cell_nodes), length(mesh.face_nodes),
        length(mesh.node_cells), length(mesh.cell_faces)))

    # cells
    for a ∈ (mesh.cell_centre, mesh.cell_volume, mesh.cell_nodes_range, mesh.cell_faces_range)
        _permute!(a, perm, done)
    end
    _reorder_lists!(mesh.cell_nodes, mesh.cell_nodes_range, buffer, n -> inperm[n])
    cell_nsign = mesh.cell_nsign
    _reorder_lists!(mesh.cell_faces, cell_faces_range, buffer, f -> ifperm[f], cell_nsign, copy(cell_nsign))
    for r ∈ cell_faces_range
        _sort_cell_faces!(mesh.cell_faces, cell_nsign, r)
    end

    # faces
    for a ∈ (mesh.face_nodes_range, face_ownerCells, mesh.face_centre, mesh.face_normal, mesh.face_e,
            mesh.face_area, mesh.face_delta, mesh.face_weight, mesh.face_gDiff)
        _permute!(a, fperm, done)
    end
    for f ∈ eachindex(face_ownerCells)
        o = face_ownerCells[f]
        face_ownerCells[f] = typeof(o)(iperm[o[1]], iperm[o[2]])
    end
    _reorder_lists!(mesh.face_nodes, mesh.face_nodes_range, buffer, n -> inperm[n])
    for f ∈ 1:nbfaces
        mesh.boundary_cellsID[f] = face_ownerCells[f][1]
    end
    # each neighbour is the other cell of the (reordered) face
    for (i, r) ∈ enumerate(cell_faces_range), j ∈ r
        o = face_ownerCells[mesh.cell_faces[j]]
        cell_neighbours[j] = o[1] == i ? o[2] : o[1]
    end

    # nodes
    _permute!(mesh.node_coords, nperm, done)
    _permute!(mesh.node_cells_range, nperm, done)
    _reorder_lists!(mesh.node_cells, mesh.node_cells_range, buffer, c -> iperm[c])
    for r ∈ mesh.node_cells_range
        sort!(view(mesh.node_cells, r))
    end
    perm
end

# a new order must cut the mean neighbour index distance by 10%, so an ordered mesh (e.g. one
# written after reordering) is read back unchanged
function _shortens_neighbour_distance(owners, nbfaces, iperm)
    old, new = 0.0, 0.0
    for f ∈ (nbfaces + 1):length(owners)
        o = owners[f]
        old += abs(Int(o[1]) - Int(o[2]))
        new += abs(Int(iperm[o[1]]) - Int(iperm[o[2]]))
    end
    new < 0.9*old
end

@inline _face_key(iperm, o) = (min(iperm[o[1]], iperm[o[2]]), max(iperm[o[1]], iperm[o[2]]))

function _invperm(perm)
    iperm = similar(perm)
    for (i, p) ∈ enumerate(perm)
        iperm[p] = i
    end
    iperm
end

# a[i] = a_old[perm[i]] by following cycles; `done` is scratch of at least length(a)
function _permute!(a, perm, done)
    fill!(done, false)
    for i ∈ eachindex(perm)
        done[i] && continue
        tmp = a[i]
        j = i
        while true
            done[j] = true
            k = perm[j]
            k == i && (a[j] = tmp; break)
            a[j] = a[k]
            j = k
        end
    end
    a
end

# rewrites list (and a companion list) so entry i holds the old entries of ranges[i] (already
# permuted), mapped by f; ranges are updated to the new positions
function _reorder_lists!(list, ranges, buffer, f, companion=nothing, companion_buffer=nothing)
    n = length(list)
    copyto!(buffer, 1, list, 1, n)
    k = 0
    for i ∈ eachindex(ranges)
        r = ranges[i]
        for j ∈ r
            k += 1
            list[k] = f(buffer[j])
            isnothing(companion) || (companion[k] = companion_buffer[j])
        end
        ranges[i] = typeof(r)(k - length(r) + 1, k)
    end
    list
end

# insertion sort of one cell's faces by face ID, moving the signs along
function _sort_cell_faces!(faces, nsign, r)
    for i ∈ (first(r) + 1):last(r)
        f, s = faces[i], nsign[i]
        j = i - 1
        while j >= first(r) && faces[j] > f
            faces[j + 1], nsign[j + 1] = faces[j], nsign[j]
            j -= 1
        end
        faces[j + 1], nsign[j + 1] = f, s
    end
end

# NEW SECTION: orderings

# neighbours of u are colval[adjacency(u)]; perm, level and queue are the only n-sized arrays
function _rcm(n, adjacency, colval, ::Type{TI}) where TI
    perm = Vector{TI}(undef, n)
    level = fill(TI(-1), n)
    queue = Vector{TI}(undef, n)
    visited = falses(n)
    degree(v) = length(adjacency(v))
    tail = 0
    for seed ∈ 1:n
        visited[seed] && continue
        start = _peripheral(seed, adjacency, colval, visited, level, queue)
        head = tail + 1
        visited[start] = true
        perm[tail += 1] = start
        while head <= tail
            u = perm[head]
            head += 1
            first_new = tail + 1
            for k ∈ adjacency(u)
                v = colval[k]
                visited[v] && continue
                visited[v] = true
                perm[tail += 1] = v
            end
            sort!(view(perm, first_new:tail), by = degree)
        end
    end
    reverse!(perm)
end

# last node reached by two breadth-first sweeps: a cheap pseudo-peripheral start
function _peripheral(seed, adjacency, colval, visited, level, queue)
    start = seed
    for _ ∈ 1:2
        queue[1] = start
        level[start] = 0
        head, tail = 1, 1
        while head <= tail
            u = queue[head]
            head += 1
            for k ∈ adjacency(u)
                v = colval[k]
                (visited[v] || level[v] >= 0) && continue
                level[v] = level[u] + 1
                queue[tail += 1] = v
            end
        end
        start = queue[tail]
        for i ∈ 1:tail
            level[queue[i]] = -1
        end
    end
    start
end

function _morton_permutation(centres, ::Type{TI}) where TI
    lo = reduce((a, b) -> min.(a, b), centres)
    hi = reduce((a, b) -> max.(a, b), centres)
    scale = (2^21 - 1)./max.(hi - lo, eps(eltype(lo)))
    keys = map(centres) do c
        q = UInt64.(floor.(Int, (c - lo).*scale))
        _spread(q[1]) | (_spread(q[2]) << 1) | (_spread(q[3]) << 2)
    end
    perm = collect(TI(1):TI(length(centres)))
    sort!(perm, by = i -> keys[i])
end

# bits of x (21 used) moved to every third position
function _spread(x::UInt64)
    x &= 0x1fffff
    x = (x | x << 32) & 0x1f00000000ffff
    x = (x | x << 16) & 0x1f0000ff0000ff
    x = (x | x << 8) & 0x100f00f00f00f00f
    x = (x | x << 4) & 0x10c30c30c30c30c3
    (x | x << 2) & 0x1249249249249249
end
