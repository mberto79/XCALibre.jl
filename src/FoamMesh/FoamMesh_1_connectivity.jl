export connect_mesh

function connect_mesh(foamdata, TI, TF)


    # OpenFOAM provides face information: sort out face connectivity first
    
    cell_nfaces = connect_cell_faces(foamdata, TI, TF)

    (
        out1 = cell_nfaces,
    )
end

function connect_cell_faces(foamdata, TI, TF)
    (;n_cells, n_ifaces, n_bfaces, faces) = foamdata 
    cell_nfaces = zeros(TI, n_cells)

    # find number of internal faces per cell
    for fi ∈ 1:n_ifaces
        face = faces[fi]
        ownerID = face.owner
        neighbourID = face.neighbour
        cell_nfaces[ownerID] += one(TI)
        cell_nfaces[neighbourID] += one(TI)
    end

    cell_faces = Vector{TI}[TI[] for _ ∈ 1:n_cells]

    for fi ∈ 1:n_ifaces 
        face = faces[fi]
        ownerID = face.owner
        neighbourID = face.neighbour
        fID = fi + n_bfaces # fID is shifted to later move boundary faces to the start
        push!(cell_faces[ownerID], fID)
        push!(cell_faces[neighbourID], fID)
    end

    

    return cell_faces
end

