
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

    cell_facesIDs = Vector{TI}[TI[] for _ ∈ 1:n_cells]
    cell_normalSign = Vector{TI}[TI[] for _ ∈ 1:n_cells]
    cell_neighbourCells = Vector{TI}[TI[] for _ ∈ 1:n_cells]

    # collect internal faces IDs, neighbours and normal signs for all cells
    for fi ∈ 1:n_ifaces 
        face = faces[fi]
        ownerID = face.owner
        neighbourID = face.neighbour
        fID = fi + n_bfaces # fID is shifted to accommodate boundary faces
        push!(cell_facesIDs[ownerID], fID)
        push!(cell_facesIDs[neighbourID], fID)

        push!(cell_normalSign[ownerID], one(TI)) # positive by definition
        push!(cell_normalSign[neighbourID], -one(TI))

        push!(cell_neighbourCells[ownerID], neighbourID)
        push!(cell_neighbourCells[neighbourID], ownerID)
    end

    n_faceIDs = sum(length.(cell_facesIDs))
    cell_faces = zeros(TI, n_faceIDs)
    cell_nsign = zeros(TI, n_faceIDs)
    cell_neighbours = zeros(TI, n_faceIDs)
    cell_faces_range = UnitRange{TI}[UnitRange{TI}(0,0) for _ ∈ 1:n_cells]

    # write to single arrays and define access ranges as UnitRange
    facei = 0 # face counter (not ID)
    for cID ∈ eachindex(cell_facesIDs)
        facesIDs = cell_facesIDs[cID]
        neighbourCells = cell_neighbourCells[cID]
        normalSign = cell_normalSign[cID]

        startIdx = TI(facei)
        for i ∈ eachindex(facesIDs, neighbourCells, normalSign)
            facei += 1
            cell_faces[facei] = facesIDs[i]
            cell_neighbours[facei] = neighbourCells[i]
            cell_nsign[facei] = normalSign[i]
        end
        endIdx = TI(facei)
        cell_faces_range[cID] = UnitRange{TI}(startIdx + one(TI), endIdx)
    end

    return cell_faces, cell_faces_range, cell_neighbours, cell_nsign
end

function connect_cell_nodes(foamdata, TI, TF)
    (;n_cells, n_ifaces, n_bfaces, faces) = foamdata 

    cell_nodesIDs = Vector{TI}[TI[] for _ ∈ 1:n_cells]

    # collect nodes IDs for internal faces for all cells
    for face ∈ faces
        ownerID = face.owner
        neighbourID = face.neighbour
        # Use fi to index here since using face data load from foamMesh
        push!(cell_nodesIDs[ownerID], face.nodesID...)
        push!(cell_nodesIDs[neighbourID], face.nodesID...)
    end
    cell_nodesIDs = intersect.(cell_nodesIDs)

    # define cell nodesIDs access ranges 
    cell_nodes_range = UnitRange{TI}[UnitRange{TI}(0,0) for _ ∈ 1:n_cells]

    startIndex = 1
    endIndex = 0
    for (cID, nodesIDs) ∈ enumerate(cell_nodesIDs)
        n_nodeIDs = length(nodesIDs)
        endIndex = startIndex + n_nodeIDs - one(TI)
        cell_nodes_range[cID] = UnitRange{TI}(startIndex, endIndex)
        startIndex += n_nodeIDs #- one(TI)
    end

    # flatten cell_nodesIDs to cell_nodes
    cell_nodes = reduce(vcat, cell_nodesIDs)

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
    (; points) = foamdata
    n_points = length(points)
    node_cellsID = Vector{TI}[TI[] for _ ∈ 1:n_points]

    # collect cell IDs for each node
    for (cID, node_range) ∈ enumerate(cell_nodes_range)
        for ID ∈ @view cell_nodes[node_range]
            push!(node_cellsID[ID], cID)
        end 
    end

    # create array range to access cell IDs 
    node_cells_range = UnitRange{TI}[0:0 for _ ∈ 1:n_points]

    startIndex = one(TI)
    endIndex = zero(TI)
    for (nID, cellsID) ∈ enumerate(node_cellsID)
        n_cells = length(cellsID)
        endIndex = startIndex + n_cells - one(TI)
        node_cells_range[nID] = UnitRange{TI}(startIndex, endIndex)
        startIndex += n_cells
    end

    node_cells = reduce(vcat, node_cellsID) # flatten array

    return node_cells, node_cells_range
end