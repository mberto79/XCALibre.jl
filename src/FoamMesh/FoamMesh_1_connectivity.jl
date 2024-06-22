export connect_mesh

function connect_mesh(points, face_nodes, face_neighbour_cell, face_owner_cell, bnames, bnFaces, bstartFace, TI, TF)

    nbfaces = sum(bnFaces)
    nifaces = minimum(bstartFace) - one(TI)

    # OpenFOAM provides face information: sort out face connectivity first

    face_owners = connect_face_owners(
        face_owner_cell, face_neighbour_cell, nbfaces, nifaces, TI, TF
        )
    
    nfaceIDs = connect_cell_faces(face_owners, face_owner_cell, face_neighbour_cell, nifaces, nbfaces, TI)

    (
        face_owners=face_owners,
        face_neighbour_cell=nbfaces, 
        face_owner_cell=nfaceIDs
    )
end

function connect_cell_faces(
    face_owners, face_owner_cell, face_neighbour_cell, nifaces, nbfaces, TI
    )
    ncells = maximum(face_owner_cell)
    cell_nfaces = zeros(TI, ncells)

    # find array size to hold faceIDs and store internal face count for all cells
    nfaceIDs = 0
    for celli ∈ 1:ncells
        nfaces1 = count(==(celli), face_neighbour_cell)
        nfaces2 = count(==(celli), @view face_owner_cell[1:nifaces])
        nfaces = nfaces1 + nfaces2
        cell_nfaces[celli] = nfaces
        nfaceIDs += nfaces
    end

    cell_faces = zeros(TI, nfaceIDs)
    fcounter = 0
    for celli ∈ 1:ncells
        for fi ∈ 1:nifaces
            cID1 = face_neighbour_cell[fi]
            cID2 = face_owner_cell[fi]
            if cID1 == celli
                fcounter += 1
                fID = fi + nbfaces
                cell_faces[fcounter] = fID
            elseif cID2 == celli
                fcounter += 1
                fID = fi + nbfaces
                cell_faces[fcounter] = fID
            end
        end
    end
    cell_faces
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