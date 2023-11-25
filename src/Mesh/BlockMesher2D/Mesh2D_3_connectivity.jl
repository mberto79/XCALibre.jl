export connect!

function connect!(mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}) where {I,F}
    assign_cellsID_to_boundary_faces!(mesh, builder)
    assign_cellsID_to_boundaries!(mesh, builder)
    assign_nodesID_to_boundaries!(mesh, builder)
    assign_cellsID_to_baffle_faces!(mesh, builder) # to do: remove allocations
    assign_cellsID_to_internal_faces!(mesh, builder)
    assign_facesID_to_cells!(mesh, builder) # allocations due to growing facesID vec
    assign_neighbours_to_cells!(mesh, builder) # allocations to grow "neighbours"
    builder = nothing
    # mesh
    nothing
end

function assign_cellsID_to_boundary_faces!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}) where {I,F}
    (; blocks) = builder
    (; faces) = mesh
    for block ∈ blocks
        (; elementsID, facesID_NS, facesID_EW) = block
        for access ∈ ((1, 1), (block.ny+1, block.ny)) # to index facesID_NS matrix
            for (row, faceID) ∈ enumerate(@view facesID_NS[:,access[1]])
                face = faces[faceID]
                elementi = elementsID[row,access[2]]
                faces[faceID] = @set face.ownerCells = SVector(elementi, elementi)
            end
        end
        for access ∈ ((1, 1), (block.nx+1, block.nx)) # to index facesID_EW matrix
            for (col, faceID) ∈ enumerate(@view facesID_EW[access[1],:])
                face = faces[faceID]
                elementi = elementsID[access[2], col]
                faces[faceID] = @set face.ownerCells = SVector(elementi, elementi)
            end
        end
    end
end

function assign_cellsID_to_baffle_faces!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}) where {I,F}
    nothing
    (; blocks, edges) = builder
    (; faces) = mesh
    # blockPair = fill(Block(zero(I)), 2)
    blockPair = [Block(zero(I)) for _ ∈ 1:2]
    edgeIndexPair = zeros(I,2)
    for (edgei, edge) ∈ enumerate(edges)
        if !edge.boundary
            find_edge_in_blocks!(blockPair, edgeIndexPair, blocks, edgei)
            if edgeIndexPair == [3, 4]
                facesID = @view blockPair[1].facesID_EW[1,:]
                ownersID1 = @view blockPair[1].elementsID[1,:]      # edge #3 -> 1
                ownersID2 = @view blockPair[2].elementsID[end,:]    # edge #4 -> end
                for (i, ID) ∈ enumerate(facesID)
                    face = faces[ID]
                    faces[ID] = @set face.ownerCells = SVector(ownersID1[i], ownersID2[i])
                end
            end
            if edgeIndexPair == [4, 3]
                facesID = @view blockPair[1].facesID_EW[end,:]
                ownersID1 = @view blockPair[1].elementsID[end,:]  # edge #4 -> end
                ownersID2 = @view blockPair[2].elementsID[1,:]    # edge #3 -> 1
                for (i, ID) ∈ enumerate(facesID)
                    face = faces[ID]
                    faces[ID] = @set face.ownerCells = SVector(ownersID1[i], ownersID2[i])
                end
            end
            if edgeIndexPair == [1, 2]
                facesID = @view blockPair[1].facesID_NS[:,1]
                ownersID1 = @view blockPair[1].elementsID[:,1]      # edge #1 -> 1
                ownersID2 = @view blockPair[2].elementsID[:,end]    # edge #2 -> end
                for (i, ID) ∈ enumerate(facesID)
                    face = faces[ID]
                    faces[ID] = @set face.ownerCells = SVector(ownersID1[i], ownersID2[i])
                end
            end
            if edgeIndexPair == [2, 1]
                facesID = @view blockPair[1].facesID_NS[:,end]
                ownersID1 = @view blockPair[1].elementsID[:,end]    # edge #2 -> end
                ownersID2 = @view blockPair[2].elementsID[:,1]      # edge #1 -> 1
                for (i, ID) ∈ enumerate(facesID)
                    face = faces[ID]
                    faces[ID] = @set face.ownerCells = SVector(ownersID1[i], ownersID2[i])
                end
            end
        end
    end
end

function assign_cellsID_to_boundaries!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}) where {I,F}
    (; faces, boundaries) = mesh
    for boundary ∈ boundaries
        (; cellsID, facesID) = boundary
        for (fi, faceID) ∈ enumerate(facesID)
            cellsID[fi] = faces[faceID].ownerCells[1]
        end
    end
end

function assign_nodesID_to_boundaries!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}) where {I,F}
    (; faces, boundaries) = mesh
    # for boundary ∈ boundaries
    #     (; nodesID, facesID) = boundary
    #     nodesID[1] = faces[facesID[1]].nodesID[1]
    #     for (fi, faceID) ∈ enumerate(facesID)
    #         nodesID[fi+1] = faces[faceID].nodesID[2]
    #     end
    # end
    for boundary ∈ boundaries 
        (; nodesID, facesID) = boundary
        for i ∈ eachindex(facesID)
            fID = facesID[i]
            face = faces[fID] 
            nodesID[i] .= face.nodesID
        end
    end
end

function assign_cellsID_to_internal_faces!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}) where {I,F}
    (; blocks) = builder
    (; faces) = mesh
    for block ∈ blocks
        (; elementsID, facesID_NS, facesID_EW, nx, ny) = block
        # process NS faces
        for j ∈ 2:(ny+1)-1
            for i ∈ 1:nx
                faceID = facesID_NS[i,j]
                face = faces[faceID]
                owner1 = elementsID[i,j-1]
                owner2 = elementsID[i,j]
                faces[faceID] = @set face.ownerCells = SVector(owner1, owner2)
            end
        end
        # process EW faces
        for j ∈ 1:ny
            for i ∈ 2:(nx+1)-1
                faceID = facesID_EW[i,j]
                face = faces[faceID]
                owner1 = elementsID[i-1,j]
                owner2 = elementsID[i,j]
                faces[faceID] = @set face.ownerCells = SVector(owner1, owner2)
            end
        end
    end
end

function assign_facesID_to_cells!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}) where {I,F}
    (; blocks) = builder
    (; cells) = mesh
    nbfaces = total_boundary_faces(mesh)
    SENW = zeros(I,4) # hold faceID for south, east, north and west faces
    for block ∈ blocks
        (; elementsID, facesID_NS, facesID_EW, nx, ny) = block
        for j ∈ 1:ny
            for i ∈ 1:nx
                cellID = elementsID[i,j]
                # cell = cells[cellID]
                SENW[1] = facesID_NS[i,j]
                SENW[2] = facesID_EW[i+1,j]
                SENW[3] = facesID_NS[i,j+1]
                SENW[4] = facesID_EW[i,j]
                for ID ∈ SENW
                    if ID > nbfaces
                        push!(cells[cellID].facesID, ID)
                    end
                end
                SENW .= zero(I)
            end
        end
    end
end

function assign_neighbours_to_cells!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}) where {I,F}
    (; cells, faces) = mesh
    for (cellID, cell) ∈ enumerate(cells)
        # for (facei, faceID) ∈ enumerate(cell.facesID)
        for faceID ∈ cell.facesID
            face = faces[faceID]
            ownerCells = face.ownerCells
            # cells[cellID].neighbours[facei] = (ownerCells .!= cellID) ⋅ ownerCells
            push!(cells[cellID].neighbours, (ownerCells .!= cellID) ⋅ ownerCells)
        end
    end
end
