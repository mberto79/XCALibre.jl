function connect_mesh(points, face_nodes, face_nodes_range, face_neighbour_cell, face_owner_cell, bnames, bnFaces, bstartFace, TI, TF)

    nbfaces = sum(bnFaces)
    nifaces = minimum(bstartFace) - one(TI)

    # OpenFOAM provides face information: sort out face connectivity first

    face_owners = connect_face_owners(
        face_owner_cell, face_neighbour_cell, nbfaces, nifaces, TI, TF
        )
    
    face_nodes, face_nodes_range = connect_face_nodes()

    (nfowners=fowners)
end

function connect_face_owners(face_owner_cell, face_neighbour_cell, nbfaces, nifaces, TI, TF)
    nfaces = length(face_owner_cell)
    zerovec = SVector{2}(zeros(TI, 2))
    face_owners = [zerovec for _ ∈ 1:nfaces]

    # loop over internal faces 
    for i ∈ eachindex(face_neighbour_cell)
        fID = i + nbfaces
        owner1 = face_owner_cell[i]
        owner2 = face_neighbour_cell[i]
        face_owners[fID] = SVector(owner1, owner2)
    end

    # loop over boundary faces
    bfaces_start = nifaces + 1
    fID = 0
    for i ∈ bfaces_start:nfaces
        fID += 1
        owner1 = face_owner_cell[i]
        owner2 = owner1
        face_owners[fID] = SVector(owner1, owner2)
    end
    face_owners
end