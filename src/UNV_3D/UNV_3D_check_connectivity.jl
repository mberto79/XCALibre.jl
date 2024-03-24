export boundary_faces
export check_face_owners
export check_cell_face_nodes
export check_node_cells
export check_all_cell_faces
export check_boundary_faces

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

function check_cell_face_nodes(mesh,cell_face_nodes)
    #Check tet cells, no. of faces=4
    #only works for meshes of same cell type
    numface=0
    (; cells,faces)=mesh
    if length(faces[1].nodes_range)==3
        numface=4
    end
    total_cell_faces=length(cells)*numface
    if length(cell_face_nodes)==total_cell_faces
        println("Passed: Length of cell_face_nodes matches calculation")
    else
        println("Failed: Warning, length of cell_face_nodes does not match calculations")
    end
end

function check_node_cells(mesh,node_cells)
    total=Int[]
    for in=1:length(mesh.nodes)
        for i=1:length(node_cells[mesh.nodes[in].cells_range])
            if findfirst(x-> x==in,mesh.cell_nodes[mesh.cells[node_cells[mesh.nodes[in].cells_range][i]].nodes_range]) !== nothing
                push!(total,1)
                break
            end
        end
    end
    if length(total)==length(mesh.nodes)
        println("Passed: Each node_cell has the correct node")
    else
        println("Failed: Error with node_cell")
    end
end

function check_all_cell_faces(mesh,all_cell_faces)
    #Check tet cells, no. of faces=4
    #only works for meshes of same cell type
    numface=0
    (; cells,faces)=mesh
    if length(faces[1].nodes_range)==3
        numface=4
    end
    total_cell_faces=length(cells)*numface
    if length(all_cell_faces)==total_cell_faces
        println("Passed: Length of all_cell_faces matches calculation")
    else
        println("Failed: Warning, length of all_cell_faces does not match calculations")
    end
end

function check_boundary_faces(boundary_cells,boundary_faces,all_cell_faces,all_cell_faces_range)
    total=[]
    for i=1:length(boundary_cells)
        bface=boundary_faces[i]
        bcell=boundary_cells[i]
        face_ID=all_cell_faces[all_cell_faces_range[bcell]]
        if findfirst(x-> x==bface,face_ID) !== nothing
            push!(total,1)
        end
    end
    if length(total)== length(boundary_faces) && length(total)==length(boundary_cells)
        println("Passed: boundary faces match cell faces")
    else
        println("Failed: boundary faces do not match cell faces")
    end
end