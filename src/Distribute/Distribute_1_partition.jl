export build_dual_graph, partition_cells, extract_subdomain, decompose, distribute

# NEW SECTION: partitioning

function build_dual_graph(mesh)
    n = length(mesh.cells)
    I = Int[]; J = Int[]
    for face ∈ mesh.faces
        o1, o2 = face.ownerCells
        if o1 != o2 # interior face
            push!(I, o1); push!(J, o2)
            push!(I, o2); push!(J, o1)
        end
    end
    sparse(I, J, ones(Int, length(I)), n, n)
end

function partition_cells(mesh, nparts::Integer)
    n = length(mesh.cells)
    nparts == 1 && return ones(Int, n)
    parts = Int.(Metis.partition(build_dual_graph(mesh), nparts; alg=:KWAY))
    counts = [count(==(r), parts) for r ∈ 1:nparts]
    cut = count(mesh.faces) do f
        o1, o2 = f.ownerCells
        o1 != o2 && parts[o1] != parts[o2]
    end
    @info "Metis partition: cells max/min = $(maximum(counts))/$(minimum(counts)), edge-cut = $cut"
    parts
end

# NEW SECTION: subdomain extraction

_mesh_like(::Mesh3, args...) = Mesh3(args...)
_mesh_like(::Mesh2, args...) = Mesh2(args...)

function extract_subdomain(mesh, parts, part::Integer)
    TI = _get_int(mesh)
    nparts = maximum(parts)
    ncells = length(mesh.cells)
    (; cells, faces, boundaries, nodes, boundary_cellsID) = mesh
    (; cell_nodes, cell_faces, cell_neighbours, cell_nsign, face_nodes) = mesh
    n_bfaces = length(boundary_cellsID)

    # owned in original order; ghosts sorted by (owning part, original id) so each
    # neighbour's ghosts form a contiguous ascending block (halo alignment invariant)
    owned = [c for c ∈ 1:ncells if parts[c] == part]
    owned_mask = falses(ncells)
    owned_mask[owned] .= true
    ghost_set = Set{Int}()
    for c ∈ owned, j ∈ cells[c].faces_range
        nb = cell_neighbours[j]
        parts[nb] != part && push!(ghost_set, nb)
    end
    ghosts = sort!(collect(ghost_set), by = g -> (parts[g], g))
    local_cells = vcat(owned, ghosts)
    n_owned, n_ghost = length(owned), length(ghosts)
    g2l = Dict{Int,TI}(c => i for (i, c) ∈ enumerate(local_cells))

    # faces: physical boundary faces first (per patch, original order), then interior
    bfaces = Int[]
    new_boundaries = eltype(boundaries)[]
    for b ∈ boundaries
        start = length(bfaces) + 1
        append!(bfaces, (fID for fID ∈ b.IDs_range if owned_mask[boundary_cellsID[fID]]))
        push!(new_boundaries, Boundary(b.name, UnitRange{TI}(start, length(bfaces))))
    end
    ifaces = Int[]
    for fID ∈ (n_bfaces+1):length(faces)
        o1, o2 = faces[fID].ownerCells
        (owned_mask[o1] || owned_mask[o2]) && push!(ifaces, fID)
    end
    local_faces = vcat(bfaces, ifaces)
    f2l = Dict{Int,TI}(f => i for (i, f) ∈ enumerate(local_faces))

    # nodes: union over local cells and faces, original order
    node_set = Set{Int}()
    for c ∈ local_cells, j ∈ cells[c].nodes_range
        push!(node_set, cell_nodes[j])
    end
    for f ∈ local_faces, j ∈ faces[f].nodes_range
        push!(node_set, face_nodes[j])
    end
    local_nodes = sort!(collect(node_set))
    n2l = Dict{Int,TI}(n => i for (i, n) ∈ enumerate(local_nodes))

    # cells: geometry copied verbatim; ghost cells keep only locally-present faces
    new_cell_nodes = TI[]; new_cell_faces = TI[]
    new_cell_neighbours = TI[]; new_cell_nsign = TI[]
    new_cells = eltype(cells)[]
    for c ∈ local_cells
        cell = cells[c]
        ns = length(new_cell_nodes) + 1
        for j ∈ cell.nodes_range
            push!(new_cell_nodes, n2l[cell_nodes[j]])
        end
        fs = length(new_cell_faces) + 1
        for j ∈ cell.faces_range
            lf = get(f2l, cell_faces[j], zero(TI))
            iszero(lf) && continue
            push!(new_cell_faces, lf)
            push!(new_cell_neighbours, g2l[cell_neighbours[j]])
            push!(new_cell_nsign, cell_nsign[j])
        end
        @reset cell.nodes_range = UnitRange{TI}(ns, length(new_cell_nodes))
        @reset cell.faces_range = UnitRange{TI}(fs, length(new_cell_faces))
        push!(new_cells, cell)
    end

    # faces: only ownerCells and nodes_range change; geometry verbatim
    new_face_nodes = TI[]
    new_faces = eltype(faces)[]
    for f ∈ local_faces
        face = faces[f]
        ns = length(new_face_nodes) + 1
        for j ∈ face.nodes_range
            push!(new_face_nodes, n2l[face_nodes[j]])
        end
        o1, o2 = face.ownerCells
        @reset face.nodes_range = UnitRange{TI}(ns, length(new_face_nodes))
        @reset face.ownerCells = SVector{2,TI}(g2l[o1], g2l[o2])
        push!(new_faces, face)
    end

    # node_cells: invert local cell_nodes
    counts = zeros(Int, length(local_nodes))
    for cell ∈ new_cells, j ∈ cell.nodes_range
        counts[new_cell_nodes[j]] += 1
    end
    stops = cumsum(counts)
    starts = stops .- counts .+ 1
    new_node_cells = zeros(TI, isempty(stops) ? 0 : stops[end])
    fill_pos = copy(starts)
    for (li, cell) ∈ enumerate(new_cells), j ∈ cell.nodes_range
        nid = new_cell_nodes[j]
        new_node_cells[fill_pos[nid]] = li
        fill_pos[nid] += 1
    end
    new_nodes = eltype(nodes)[]
    for (i, n) ∈ enumerate(local_nodes)
        node = nodes[n]
        @reset node.cells_range = UnitRange{TI}(starts[i], stops[i])
        push!(new_nodes, node)
    end

    new_boundary_cellsID = TI[g2l[boundary_cellsID[f]] for f ∈ bfaces]

    lmesh = _mesh_like(mesh,
        new_cells, new_cell_nodes, new_cell_faces, new_cell_neighbours, new_cell_nsign,
        new_faces, new_face_nodes, new_boundaries, new_nodes, new_node_cells,
        mesh.get_float, mesh.get_int, new_boundary_cellsID)

    # global block renumbering: new id = part offset + position within part (orig order)
    part_counts = [count(==(r), parts) for r ∈ 1:nparts]
    offs = cumsum(vcat(0, part_counts))
    pos = zeros(Int, ncells)
    ctr = zeros(Int, nparts)
    for c ∈ 1:ncells
        r = parts[c]
        ctr[r] += 1
        pos[c] = ctr[r]
    end
    l2g = TI[offs[parts[c]] + pos[c] for c ∈ local_cells]
    owner = TI[parts[c] - 1 for c ∈ local_cells]
    partition = Partition(part - 1, nparts, n_owned, n_ghost, l2g, owner,
        offs[part] + 1, offs[part] + part_counts[part])

    # processor patches: send/recv sorted by original global id (alignment invariant)
    procs = ProcessorPatch{Vector{TI}}[]
    for q ∈ sort!(unique(parts[g] for g ∈ ghosts))
        pfaces = TI[]
        send = Set{TI}()
        for (lf, f) ∈ enumerate(ifaces)
            o1, o2 = faces[f].ownerCells
            if owned_mask[o1] && parts[o2] == q
                push!(pfaces, length(bfaces) + lf); push!(send, g2l[o1])
            elseif owned_mask[o2] && parts[o1] == q
                push!(pfaces, length(bfaces) + lf); push!(send, g2l[o2])
            end
        end
        recv_ghosts = TI[i for i ∈ n_owned+1:n_owned+n_ghost if parts[local_cells[i]] == q]
        push!(procs, ProcessorPatch(q - 1, pfaces, sort!(collect(send)), recv_ghosts))
    end

    DistributedMesh(lmesh, partition, procs, TI.(local_cells), TI.(local_faces))
end

# NEW SECTION: entry points

# single-process decomposition (testing, offline tooling)
function decompose(mesh, nparts::Integer)
    parts = partition_cells(mesh, nparts)
    [extract_subdomain(mesh, parts, r) for r ∈ 1:nparts]
end

"""
    distribute(mesh; comm=MPI.COMM_WORLD)

Online mesh distribution: rank 0 partitions `mesh` (Metis k-way) and scatters one
`DistributedMesh` per rank; other ranks may pass `nothing` as `mesh`.
"""
function distribute(mesh; comm=MPI.COMM_WORLD)
    MPI.Initialized() || MPI.Init()
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    nranks == 1 && return extract_subdomain(mesh, partition_cells(mesh, 1), 1)
    if rank == 0
        parts = partition_cells(mesh, nranks)
        for q ∈ 1:nranks-1
            MPI.send(extract_subdomain(mesh, parts, q + 1), comm; dest=q, tag=0)
        end
        extract_subdomain(mesh, parts, 1)
    else
        MPI.recv(comm; source=0, tag=0)
    end
end
