
function connect_mesh(foamdata, TI, TF)

    cell_faces, cell_faces_range, cell_neighbours, cell_nsign = connect_cell_faces(foamdata, TI, TF)
    
    cell_nodes, cell_nodes_range = connect_cell_nodes(foamdata, TI, TF)

    node_cells, node_cells_range = connect_node_cells(
        foamdata, cell_nodes, cell_nodes_range, TI, TF
        )
        
    face_nodes, face_nodes_range = connect_face_nodes(foamdata, TI, TF)

    connectivity = (
        ; cell_faces, cell_faces_range, cell_neighbours, cell_nsign,
        cell_nodes, cell_nodes_range, 
        node_cells, node_cells_range, 
        face_nodes, face_nodes_range,
        )

    return connectivity
end

function connect_cell_faces(foamdata, TI, TF)
    (;n_cells, n_ifaces, n_bfaces, faces) = foamdata

    # CSR two-pass fill: byte-identical to push!-then-flatten ordering
    # Pass 1: degree count per cell
    count = zeros(TI, n_cells)
    for fi ∈ 1:n_ifaces
        face = faces[fi]
        count[face.owner] += one(TI)
        count[face.neighbour] += one(TI)
    end

    # build offsets and access ranges
    offset = zeros(TI, n_cells)
    cell_faces_range = UnitRange{TI}[UnitRange{TI}(0,0) for _ ∈ 1:n_cells]
    for c ∈ 1:n_cells
        offset[c] = c == 1 ? zero(TI) : offset[c-1] + count[c-1]
        cell_faces_range[c] = UnitRange{TI}(offset[c] + one(TI), offset[c] + count[c])
    end

    total = TI(2 * n_ifaces)
    cell_faces = Vector{TI}(undef, total)
    cell_nsign = Vector{TI}(undef, total)
    cell_neighbours = Vector{TI}(undef, total)

    # Pass 2: fill in increasing-fi order, owner before neighbour per face
    cursor = copy(offset)
    for fi ∈ 1:n_ifaces
        face = faces[fi]
        ownerID = face.owner
        neighbourID = face.neighbour
        fID = fi + n_bfaces # fID is shifted to accommodate boundary faces

        cursor[ownerID] += one(TI)
        k = cursor[ownerID]
        cell_faces[k] = fID
        cell_nsign[k] = one(TI) # positive by definition
        cell_neighbours[k] = neighbourID

        cursor[neighbourID] += one(TI)
        k = cursor[neighbourID]
        cell_faces[k] = fID
        cell_nsign[k] = -one(TI)
        cell_neighbours[k] = ownerID
    end

    return cell_faces, cell_faces_range, cell_neighbours, cell_nsign
end

function connect_cell_nodes(foamdata, TI, TF)
    (; n_cells, faces) = foamdata
    n_points = length(foamdata.points)

    # Pass 1: raw count (boundary faces: owner==neighbour → each node counted twice)
    rcount = zeros(TI, n_cells)
    for face ∈ faces
        nn = length(face.nodesID)
        rcount[face.owner] += TI(nn)
        rcount[face.neighbour] += TI(nn)
    end

    # raw offsets
    roff = zeros(TI, n_cells)
    for c ∈ 1:n_cells
        roff[c] = c == 1 ? zero(TI) : roff[c-1] + rcount[c-1]
    end

    raw = Vector{TI}(undef, roff[n_cells] + rcount[n_cells])
    rcur = copy(roff)

    # Pass 2: fill raw (owner role then neighbour role per face, global face order)
    for face ∈ faces
        nodes = face.nodesID
        o = face.owner
        for nid ∈ nodes
            rcur[o] += one(TI)
            raw[rcur[o]] = nid
        end
        nb = face.neighbour
        for nid ∈ nodes
            rcur[nb] += one(TI)
            raw[rcur[nb]] = nid
        end
    end

    # Pass 3a: dedup count per cell using stamp (0 never collides with cell id ≥ 1)
    seen = zeros(TI, n_points)
    dcount = zeros(TI, n_cells)
    for c ∈ 1:n_cells
        for k ∈ (roff[c]+one(TI)):(roff[c]+rcount[c])
            nid = raw[k]
            if seen[nid] != TI(c)
                dcount[c] += one(TI)
                seen[nid] = TI(c)
            end
        end
    end

    # build dedup offsets and ranges
    doff = zeros(TI, n_cells)
    cell_nodes_range = Vector{UnitRange{TI}}(undef, n_cells)
    for c ∈ 1:n_cells
        doff[c] = c == 1 ? zero(TI) : doff[c-1] + dcount[c-1]
        cell_nodes_range[c] = UnitRange{TI}(doff[c]+one(TI), doff[c]+dcount[c])
    end
    cell_nodes = Vector{TI}(undef, doff[n_cells] + dcount[n_cells])

    # Pass 3b: dedup fill (reset stamp then refill)
    fill!(seen, zero(TI))
    dcur = copy(doff)
    for c ∈ 1:n_cells
        for k ∈ (roff[c]+one(TI)):(roff[c]+rcount[c])
            nid = raw[k]
            if seen[nid] != TI(c)
                dcur[c] += one(TI)
                cell_nodes[dcur[c]] = nid
                seen[nid] = TI(c)
            end
        end
    end

    return cell_nodes, cell_nodes_range
end

function connect_face_nodes(foamdata, TI, TF)
    (; n_ifaces, n_faces, faces) = foamdata

    nFaceNodes = 0 # number of nodes to store
    for face ∈ faces
        nFaceNodes += length(face.nodesID)
    end
    
    face_nodes = zeros(TI, nFaceNodes)
    face_nodes_range = UnitRange{TI}[0:0 for _ ∈ 1:n_faces]

    # assign boundary faces nodesIDs first
    nodei = 0
    startIndex = one(TI)
    endIndex = zero(TI)
    fID = zero(TI) # actual face ID to use
    for bfacei ∈ (n_ifaces + 1):n_faces # careful: use bfacei to index foam faces
        fID += one(TI)
        face = faces[bfacei]
        nodesID = face.nodesID
        # assign access range
        n_nodes = length(nodesID)
        endIndex = startIndex + n_nodes - one(TI)
        face_nodes_range[fID] = UnitRange{TI}(startIndex, endIndex)
        startIndex += n_nodes
        # assign actual node IDs values to long array
        for nID ∈ nodesID
            nodei += 1
            face_nodes[nodei] = nID
        end
    end

    # now assign internal faces nodesIDs
    for bfacei ∈ 1:n_ifaces # careful: use bfacei to index foam faces
        fID += one(TI)
        face = faces[bfacei]
        nodesID = face.nodesID
        # assign access range
        n_nodes = length(nodesID)
        endIndex = startIndex + n_nodes - one(TI)
        face_nodes_range[fID] = UnitRange{TI}(startIndex, endIndex)
        startIndex += n_nodes
        # assign actual node IDs values to long array
        for nID ∈ face.nodesID
            nodei += 1
            face_nodes[nodei] = nID
        end
    end

    return face_nodes, face_nodes_range
end

function connect_node_cells(foamdata, cell_nodes, cell_nodes_range, TI, TF)
    n_points = length(foamdata.points)
    n_cells = length(cell_nodes_range)

    # Pass 1: count cells per node (ascending cID → ascending cell list per node)
    count = zeros(TI, n_points)
    for cID ∈ 1:n_cells
        for ID ∈ @view cell_nodes[cell_nodes_range[cID]]
            count[ID] += one(TI)
        end
    end

    # build offsets and ranges
    offset = zeros(TI, n_points)
    node_cells_range = Vector{UnitRange{TI}}(undef, n_points)
    for n ∈ 1:n_points
        offset[n] = n == 1 ? zero(TI) : offset[n-1] + count[n-1]
        node_cells_range[n] = UnitRange{TI}(offset[n]+one(TI), offset[n]+count[n])
    end

    total = n_points == 0 ? zero(TI) : offset[n_points] + count[n_points]
    node_cells = Vector{TI}(undef, total)

    # Pass 2: fill in ascending cID order
    cursor = copy(offset)
    for cID ∈ 1:n_cells
        for ID ∈ @view cell_nodes[cell_nodes_range[cID]]
            cursor[ID] += one(TI)
            node_cells[cursor[ID]] = TI(cID)
        end
    end

    return node_cells, node_cells_range
end