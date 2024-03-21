export boundary_faces
export check_face_owners

# Function to work out number of boundary faces (based on boundary IDs_range)
function boundary_faces(mesh)
    nbfaces = 0 # number of boundary faces
    for boundary ∈ mesh.boundaries
        nbfaces = max(nbfaces, maximum(boundary.IDs_range))
    end
    return nbfaces
end

# Function to check existence of faceID in cell_faces (accessed via faces_range) 
function check_face_owners(mesh)
    (; cells, faces, cell_faces) = mesh
    test_range = (boundary_faces(mesh) + 1):length(faces) # test only internal faces
    results = [false for i ∈ test_range] # preallocate results
    faces_checked = [0 for i ∈ test_range] # IDs of internal faces checked
    for (i, face_ID) ∈ enumerate(test_range)
        face = faces[face_ID]
        ownerCells = face.ownerCells
        owner1 = ownerCells[1] # cell ID of owner 1
        owner2 = ownerCells[2] # cell ID of owner 2
        owner1_face_range = cells[owner1].faces_range
        owner2_face_range = cells[owner2].faces_range
        owner1_fIDs = @view cell_faces[owner1_face_range]
        owner2_fIDs = @view cell_faces[owner2_face_range]
        owner1_check = face_ID ∈ owner1_fIDs
        owner2_check = face_ID ∈ owner2_fIDs
        if owner1_check && owner2_check
            results[i] = true
            faces_checked[i] = face_ID
        else
            # println("Face owners not consistent for face ", face_ID)
            # face_checks[face_ID] = false # not needed array was initialised with false
        end  
    end
    passed = sum(results)
    failed = length(results) - passed
    println("Faces owners check: passed ", passed, " failed ", failed)
    faces_checked, results
end
