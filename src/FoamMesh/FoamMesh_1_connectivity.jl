export connect_mesh

function connect_mesh(foamdata, TI, TF)

    
    # OpenFOAM provides face information: sort out face-cell connectivity first
    
    cell_faces, cell_faces_range, cell_neighbours, cell_nsign = connect_cell_faces(foamdata, TI, TF)
    
    cell_nodes, cell_nodes_range = connect_cell_nodes(foamdata, TI, TF)

    face_nodes, face_nodes_range = connect_face_nodes(foamdata, TI, TF) # to do
    node_cells, node_cells_range = connect_node_cells(foamdata, TI, TF) # to do
    
    boundaries = generate_boundaries(foamdata, TI, TF)

    (
        out1 = cell_faces,
        out2 = cell_faces_range,
        out3 = cell_neighbours,
        out4 = cell_nsign,
        out5 = cell_nodes,
        out6 = cell_nodes_range,
        out7 = face_nodes, 
        out8 = face_nodes_range, 
        out9 = boundaries,
    )
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
        fID = fi + n_bfaces # fID is shifted to accommodate boundary faces at the start
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

    # write to single arrays and define equivalent access ranges as UnitRange
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
    n_nodeIDs = sum(length.(cell_nodesIDs))
    cell_nodes_range = UnitRange{TI}[UnitRange{TI}(0,0) for _ ∈ 1:n_nodeIDs]

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

    nodei = 0
    for bfacei ∈ (n_ifaces + 1):n_faces # careful: use bfacei to index foam faces
        face = faces[bfacei]
        for nID ∈ face.nodesID
            nodei += 1
            face_nodes[nodei] = nID
        end
    end

    for bfacei ∈ 1:n_ifaces # careful: use bfacei to index foam faces
        face = faces[bfacei]
        for nID ∈ face.nodesID
            nodei += 1
            face_nodes[nodei] = nID
        end
    end
            


    face_nodes_range = zero(TI)
    return face_nodes, face_nodes_range
end

function connect_node_cells(foamdata, TI, TF)
    # node_cells, node_cells_range = zero(TI)
    # return node_cells, node_cells_range
    0, 0
end

function generate_boundaries(foamdata, TI, TF)
    (; n_bfaces, n_faces, n_ifaces) = foamdata
    foamBoundaries = foamdata.boundaries
    boundaries = Mesh.Boundary{Symbol,UnitRange{TI}}[
        Mesh.Boundary(:default, UnitRange{TI}(0,0)) for _ ∈ eachindex(foamBoundaries)]

    startIndex = 1
    endIndex = 0
    for (bi, foamboundary) ∈ enumerate(foamBoundaries)
        (; name, nFaces) = foamboundary
        endIndex = startIndex + nFaces - one(TI)
        boundaries[bi] = Mesh.Boundary{Symbol,UnitRange{TI}}(name, startIndex:endIndex)
        startIndex += nFaces
    end

    return boundaries
end