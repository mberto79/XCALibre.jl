
function generate_mesh(foamdata, connectivity, TI, TF) # TI and TF are int and float types
    boundaries = generate_boundaries(foamdata, TI, TF)
    nodes = Mesh.Node.(foamdata.points, connectivity.node_cells_range)
    cells = Mesh.Cell.(
        Ref(SVector{3}(zeros(Float64, 3))),
        Ref(0.0),
        connectivity.cell_nodes_range,
        connectivity.cell_faces_range
        )
    faces = generate_faces(foamdata, connectivity, TI, TF)

    boundary_cellsID = generate_boundary_cells(foamdata, boundaries, faces, TI, TF)

    return Mesh.Mesh3(
        cells,
        connectivity.cell_nodes,
        connectivity.cell_faces,
        connectivity.cell_neighbours,
        connectivity.cell_nsign,
        faces,
        connectivity.face_nodes,
        boundaries,
        nodes,
        connectivity.node_cells,
        SVector{3}(zeros(TF,3)),
        UnitRange{TI}(0,0),
        boundary_cellsID
    )
end

function generate_boundary_cells(foamdata, boundaries, faces, TI, TF)
    (; n_bfaces) = foamdata 
    boundary_cellsID = zeros(TI, n_bfaces)

    for boundary ∈ boundaries 
        for fID ∈ boundary.IDs_range
            face = faces[fID]
            boundary_cellsID[fID] = face.ownerCells[1]
        end
    end

    return boundary_cellsID
end

function generate_faces(foamdata, connectivity, TI, TF)
    (; n_faces, n_bfaces, n_ifaces) = foamdata
    OF_faces = foamdata.faces
    orderedFaces = similar(OF_faces)
    orderedFaces[1:n_bfaces] = OF_faces[(n_ifaces+1):end] # move boundary faces to start
    orderedFaces[n_bfaces+1:end] = OF_faces[1:(n_ifaces)] # shift internal faces to end
    (; face_nodes_range) = connectivity # these are correctly ordered

    dummy_face = Mesh.Face3D(TI, TF)
    faceType = typeof(dummy_face)
    faces = Vector{faceType}(undef, n_faces)

    for (fID, face) ∈ enumerate(faces)
        OFace = orderedFaces[fID]
        @reset face.nodes_range = face_nodes_range[fID]
        @reset face.ownerCells = SVector{2}(TI[OFace.owner, OFace.neighbour])
        faces[fID] = face
    end

    return faces
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