export UNV3D_mesh

"""
    UNV3D_mesh(unv_mesh; scale=1, integer_type=Int64, float_type=Float64)

Read and convert 3D UNV mesh file into XCALibre.jl. Note that a limitation of the .unv mesh format is that it only supports the following 3D cells:

- Tetahedrals
- Prisms
- Hexahedrals

### Input

- `unv_mesh` -- path to mesh file.

### Optional arguments

- `scale` -- used to scale mesh file e.g. scale=0.001 will convert mesh from mm to metres defaults to 1 i.e. no scaling

- `integer_type` - select interger type to use in the mesh (Int32 may be useful on GPU runs) 

- `float_type` - select interger type to use in the mesh (Float32 may be useful on GPU runs) 

"""
function UNV3D_mesh(unv_mesh; scale=1, integer_type=Int64, float_type=Float64)
    stats = @timed begin
        println("Loading UNV File...")
        points, efaces, cells_UNV, boundaryElements = read_UNV3( # "volumes" changed to cells_UNV
            unv_mesh; scale=scale, integer=integer_type, float=float_type)
        println("File Read Successfully")
        println("Generating Mesh...")

        cell_nodes, cell_nodes_range = generate_cell_nodes(cells_UNV) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
        node_cells, node_cells_range = generate_node_cells(points, cells_UNV)  # Should be Hybrid compatible, tested for hexa.
        nodes = build_nodes(points, node_cells_range) # Hyrbid compatible, works for Tet and Hexa
        boundaries = build_boundaries(boundaryElements) # Hybrid compatible

        nbfaces = sum(length.(getproperty.(boundaries, :IDs_range))) # total boundary faces

        bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = 
        begin
            generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, cells_UNV) # Hybrid compatible, tested with hexa
        end

        iface_nodes, iface_nodes_range, iface_owners_cells = 
        begin 
            generate_internal_faces(cells_UNV, nbfaces, nodes, node_cells) # Hybrid compatible, tested with hexa.
        end

        # NOTE: A function will be needed here to reorder the nodes IDs of "faces" to be geometrically sound! (not needed for tet cells though)
        bface_nodes,iface_nodes=order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes,nodes)
        #2 methods, using old as new function produced negative volumes?
        # Old method needs clean up

        # Shift range of nodes_range for internal faces (since it will be appended)
        iface_nodes_range .= [
            iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
            ]

        # Concatenate boundary and internal faces
        face_nodes = vcat(bface_nodes, iface_nodes)
        face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
        face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)

        # Sort out cell to face connectivity
        cell_faces, cell_nsign, cell_faces_range, cell_neighbours = begin
            generate_cell_face_connectivity(cells_UNV, nbfaces, face_owner_cells) # Hybrid compatible. Hexa and tet tested.
        end

        # Build mesh (without calculation of geometry/properties)
        cells = build_cells(cell_nodes_range, cell_faces_range) # Hybrid compatible. Hexa tested.
        faces = build_faces(face_nodes_range, face_owner_cells) # Hybrid compatible. Hexa tested.

        mesh = Mesh3(
            cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, 
            faces, face_nodes, boundaries, 
            nodes, node_cells,
            SVector{3, Float64}(0.0, 0.0, 0.0), UnitRange{Int64}(0, 0), boundary_cellsID
        ) # Hexa tested.

        # Update mesh to include all geometry calculations required
        calculate_centres!(mesh) # Uses centroid instead of geometric. Will need changing, should work fine for regular cells and faces
        calculate_face_properties!(mesh) # Touched up face properties, double check values.
        calculate_area_and_volume!(mesh) # Will only work for Tet,Hexa, Prism cells

        return mesh
    end
end

# Convenience access FUNCTIONS
get_data(array, range, index) = @view array[range[index]]
get_data(array, range) =  array[range] #@view array[range] # 
nodeIDs = get_data
faceIDs = get_data
cellIDs = get_data

#Functions for Face Node Order

segment(p1, p2) = p2 - p1
unit_vector(vec) = vec/norm(vec)
angle1(s, i1, i2) = acosd( (s[i1]⋅s[i2])/(norm(s[i1])*norm(s[i2]))) #Old
#angle1(vec1, vec2) = acosd( (vec1⋅vec2)/(norm(vec1)*norm(vec2))) # New

# BUILD mesh functions

build_cells(cell_nodes_range, cell_faces_range) = begin
    # Allocate memory for cells array
    cells = [Cell(Int64, Float64) for _ ∈ eachindex(cell_faces_range)]

    # update cell nodes and faces ranges (using Accessors.jl)
    for cID ∈ eachindex(cell_nodes_range)
        cell = cells[cID]
        @reset cell.nodes_range = cell_nodes_range[cID]
        @reset cell.faces_range = cell_faces_range[cID]
        cells[cID] = cell
    end

    return cells
end

build_faces(face_nodes_range, face_owner_cells) = begin
    # Allocate memory for faces array
    faces = [Face3D(Int64, Float64) for _ ∈ eachindex(face_nodes_range)]

    # Update face nodes range and owner cells
    for fID ∈ eachindex(face_nodes_range)
        face = faces[fID]
        @reset face.nodes_range = face_nodes_range[fID]
        @reset face.ownerCells = SVector{2,Int64}(face_owner_cells[fID])
        faces[fID] = face # needed to replace entry with new Face3D object
    end

    return faces
end

function build_nodes(points, node_cells_range) # Hybrid compatible. Works for Tet and Hexa
    nodes = [Node(SVector{3, Float64}(0.0,0.0,0.0), 1:1) for _ ∈ eachindex(points)]
    @inbounds for i ∈ eachindex(points)
        nodes[i] =  Node(points[i].xyz, node_cells_range[i])
    end
    return nodes
end

function build_boundaries(boundaryElements)
    bfaces_start = 1
    boundaries = Vector{Boundary{Symbol,UnitRange{Int64}}}(undef,length(boundaryElements))
    for (i, boundaryElement) ∈ enumerate(boundaryElements)
        bfaces = length(boundaryElement.facesID)
        bfaces_range = UnitRange{Int64}(bfaces_start:(bfaces_start + bfaces - 1))
        boundaries[i] = Boundary(Symbol(boundaryElement.name), bfaces_range)
        bfaces_start += bfaces
    end
    return boundaries
end

UNV2D_mesh() = begin # Not needed?
    nothing
    # return mesh
end

# GENERATE Functions

function generate_cell_nodes(cells_UNV)
    cell_nodes = Int64[] # cell_node length is undetermined as mesh could be hybrid, using push. Could use for and if before to preallocate vector.
    
    # Note 0: this errors with prism and hex for some reason? Reader? (volumentCount !- length(volumes))
    # NOTE 1: You could also run a loop over all the "volumes" and accumulate their size. Then allocate
    # NOTE 2: Or you could allocate an empty vector of vector of size ncells (less performant than NOTE 1 but faster than the current method)
    for n = eachindex(cells_UNV)
        for i = 1:cells_UNV[n].nodeCount
            push!(cell_nodes,cells_UNV[n].nodesID[i])
        end
    end

    cell_nodes_range = Vector{UnitRange{Int64}}(undef, length(cells_UNV)) # cell_nodes_range determined by no. of cells.
    x = 0
    for i = eachindex(cells_UNV)
        cell_nodes_range[i] = UnitRange(x + 1, x + length(cells_UNV[i].nodesID))
        x = x + length(cells_UNV[i].nodesID)
    end
    return cell_nodes, cell_nodes_range
end

function generate_node_cells(points, cells_UNV)
    temp_node_cells = [Int64[] for _ ∈ eachindex(points)] # array of vectors to hold cellIDs

    # Add cellID to each point that defines a "volume"
    for (cellID, cell) ∈ enumerate(cells_UNV)
        for nodeID ∈ cell.nodesID
            push!(temp_node_cells[nodeID], cellID)
        end
    end # Should be hybrid compatible 

    node_cells_size = sum(length.(temp_node_cells)) # number of cells in node_cells

    node_cells_index = 0 # change to node cells index
    node_cells = zeros(Int64, node_cells_size)
    node_cells_range = [UnitRange{Int64}(1, 1) for _ ∈ eachindex(points)]
    for (nodeID, cellsID) ∈ enumerate(temp_node_cells)
        for cellID ∈ cellsID
            node_cells_index += 1
            node_cells[node_cells_index] = cellID
        end
        node_cells_range[nodeID] = UnitRange{Int64}(node_cells_index - length(cellsID) + 1, node_cells_index)
    end
    return node_cells, node_cells_range
end #Tested for hexa cells, working

function generate_boundary_faces(
    boundaryElements, efaces, nbfaces, node_cells, node_cells_range, cells_UNV
    )
    bface_nodes = Vector{Vector{Int64}}(undef, nbfaces)
    bface_nodes_range = Vector{UnitRange{Int64}}(undef, nbfaces)
    bowners_cells = Vector{Int64}[Int64[0,0] for _ ∈ 1:nbfaces]
    boundary_cells = Vector{Int64}(undef, nbfaces)

    fID = 0 # faceID index of output array (reordered)
    start = 1
    for boundary ∈ boundaryElements
        elements = boundary.facesID
            for bfaceID ∈ elements
                fID += 1
                nnodes = length(efaces[bfaceID].nodesID)
                nodeIDs = efaces[bfaceID].nodesID # Actually nodesIDs
                bface_nodes[fID] = nodeIDs
                bface_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
                start += nnodes

                # Find owner cells (same as boundary cells)
                assigned = false
                for nodeID ∈ nodeIDs
                    cIDs = cellIDs(node_cells, node_cells_range, nodeID)
                    for cID ∈ cIDs
                        if intersect(nodeIDs, cells_UNV[cID].nodesID) == nodeIDs
                            bowners_cells[fID] .= cID
                            boundary_cells[fID] = cID
                            assigned = true
                            break
                        end
                    end
                    if assigned 
                        break
                    end
                end
            end
    end

    bface_nodes = vcat(bface_nodes...) # Flatten - from Vector{Vector{Int}} to Vector{Int}
    # Check: Length of bface_nodes = no. of bfaces x no. of nodes of each bface

    return bface_nodes, bface_nodes_range, bowners_cells, boundary_cells
end

function generate_internal_faces(cells_UNV, nbfaces, nodes, node_cells)
    # determine total number of faces based on cell type (including duplicates)
    total_faces = 0
    for cell ∈ cells_UNV
        # add faces for tets
        if cell.nodeCount == 4
            total_faces += 4
        end
        # add faces for Hexa
        if cell.nodeCount == 8 
            total_faces += 6
        end
        #add faces for Wedge/Penta
        if cell.nodeCount == 6
            total_faces += 5
        end
    end

    # Face nodeIDs for each cell is a vector of vectors of vectors :-)
    cells_faces_nodeIDs = Vector{Vector{Int64}}[Vector{Int64}[] for _ ∈ 1:length(cells_UNV)] 

    # Generate all faces for each cell/element/volume
    for (cellID, cell) ∈ enumerate(cells_UNV)
        # Generate faces for tet elements
        if cell.nodeCount == 4
            nodesID = cell.nodesID
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[4]])
        end
        # add conditions for other cell types
        # Generate faces for hexa elements using UNV structure method
        if cell.nodeCount == 8
            nodesID = cell.nodesID
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[5], nodesID[6], nodesID[7], nodesID[8]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[5], nodesID[6]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[3], nodesID[4], nodesID[7], nodesID[8]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[6], nodesID[7]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[4], nodesID[5], nodesID[8]])
        end
        # Generate faces for prism elements, using pattern in UNV file.
        if cell.nodeCount == 6
            nodesID = cell.nodesID
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]]) # Triangle 1
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[4], nodesID[5], nodesID[6]]) # Triangle 2
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[4], nodesID[5]]) # Rectangle 1
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[5], nodesID[6]]) # Rectangle 2
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[3], nodesID[4], nodesID[6]]) # Rectangle 3
        end
    end

    # Sort nodesIDs for each face based on ID (need to correct order later to be physical)
    # this allows to find duplicates later more easily (query on ordered ids is faster)
    for face_nodesID ∈ cells_faces_nodeIDs
        for nodesID ∈ face_nodesID
            sort!(nodesID)
        end
    end

    # Find owner cells for each face
    owners_cellIDs = Vector{Int64}[zeros(Int64, 2) for _ ∈ 1:total_faces]
    facei = 0 # faceID counter (will be reduced to internal faces later)
    for (cellID, faces_nodeIDs) ∈ enumerate(cells_faces_nodeIDs) # loop over cells
        for facei_nodeIDs ∈ faces_nodeIDs # loop over vectors of nodeIDs for each face
            facei += 1 # face counter
            owners_cellIDs[facei][1] = cellID # ID of first cell containing the face
            for nodeID ∈ facei_nodeIDs # loop over ID of each node in the face
                cells_range = nodes[nodeID].cells_range
                node_cellIDs = @view node_cells[cells_range] # find cells that use this node
                for nodei_cellID ∈ node_cellIDs # loop over cells that share the face node
                    if nodei_cellID !== cellID # ensure cellID is not same as current cell 
                        for face ∈ cells_faces_nodeIDs[nodei_cellID]
                            if face == facei_nodeIDs
                                owners_cellIDs[facei][2] = nodei_cellID # set owner cell ID 
                                break
                            end
                        end
                    end
                end
            end
        end
    end

    # Sort all face owner vectors
    sort!.(owners_cellIDs) # in-place sorting

    # Extract nodesIDs for each face from all cells into a vector of vectors
    face_nodes = Vector{Int64}[Int64[] for _ ∈ 1:total_faces] # nodesID for all faces
    fID = 0 # counter to keep track of faceID
    for celli_faces_nodeIDs ∈ cells_faces_nodeIDs
        for nodesID ∈ celli_faces_nodeIDs
            fID += 1
            face_nodes[fID] = nodesID
        end
    end

    # Remove duplicates
    unique_indices = unique(i -> face_nodes[i], eachindex(face_nodes))
    unique!(face_nodes)
    keepat!(owners_cellIDs, unique_indices)

    # Remove boundary faces

    total_bfaces = 0 # count boundary faces
    for owners ∈ owners_cellIDs
        if owners[1] == 0
            total_bfaces += 1
        end
    end

    bfaces_indices = zeros(Int64, total_bfaces) # preallocate memory
    counter = 0
    for (i, owners) ∈ enumerate(owners_cellIDs)
        if owners[1] == 0
            counter += 1
            bfaces_indices[counter] = i # contains indices of faces to remove
        end
    end

    deleteat!(owners_cellIDs, bfaces_indices)
    deleteat!(face_nodes, bfaces_indices)

    println("Removing ", total_bfaces, " (from ", nbfaces, ") boundary faces")

    # Generate face_nodes_range
    face_nodes_range = Vector{UnitRange{Int64}}(undef, length(face_nodes))
    start = 1
    for (fID, nodesID) ∈ enumerate(face_nodes)
        nnodes = length(nodesID)
        face_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
        start += nnodes
    end

    # Flatten array i.e. go from Vector{Vector{Int}} to Vector{Int}
    face_nodes = vcat(face_nodes...) 

    return face_nodes, face_nodes_range, owners_cellIDs
end

function order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes,nodes) # Old Method

    n_bfaces = length(bface_nodes_range)
    n_ifaces =  length(iface_nodes_range)

    for fID = 1:n_bfaces # Re-order Boundary Faces
        if length(bface_nodes_range[fID]) == 4 # Only for Quad faces
            nIDs=nodeIDs(bface_nodes,bface_nodes_range[fID]) # Get ids of nodes of face
    
            ordered_ID=sort(nIDs) # sort them so that the lowest ID is first
    
            n1=nodes[ordered_ID[1]].coords # Get coords of 4 nodes
            n2=nodes[ordered_ID[2]].coords
            n3=nodes[ordered_ID[3]].coords
            n4=nodes[ordered_ID[4]].coords
    
            points = [n1, n2, n3, n4]
    
            _x(n) = n[1]
            _y(n) = n[2]
            _z(n) = n[3]
    
            l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
            fn = unit_vector(l[2] × l[3]) # face normal vector
    
            angles=Float64[] # Vector to store angles
            theta2 = 0.0
            theta3 = angle1(l, 2, 3)*(signbit((l[2] × fn)⋅l[3]) ? 1 : -1)
            theta4 = angle1(l, 2, 4)*(signbit((l[2] × fn)⋅l[4]) ? 1 : -1)
    
            push!(angles,theta2,theta3,theta4)
    
            dict=Dict() # Using dictionary to link noode IDs to angle
            for (n,f) in enumerate(angles)
                dict[f] = ordered_ID[n+1]
            end
    
            sorted_angles=sort(angles) # Sort angles from smallest to largest. Right hand rule.
    
            sorted_IDs=Int64[]
            push!(sorted_IDs,ordered_ID[1])
            push!(sorted_IDs,dict[sorted_angles[1]])
            push!(sorted_IDs,dict[sorted_angles[2]])
            push!(sorted_IDs,dict[sorted_angles[3]])
    
            counter=0
            for i=bface_nodes_range[fID] # Re-writing face_nodes with ordered nodes
                counter=counter+1
                bface_nodes[i]=sorted_IDs[counter]
            end
        end
    end
    
    for fID = 1:n_ifaces # Re-order internal faces
        if length(iface_nodes_range[fID])==4 # Only for Quad Faces
            nIDs=nodeIDs(iface_nodes,iface_nodes_range[fID]) # Get ids of nodes of the face
    
            ordered_ID=sort(nIDs) # Sort them so that the lowest ID is first
    
            n1=nodes[ordered_ID[1]].coords
            n2=nodes[ordered_ID[2]].coords
            n3=nodes[ordered_ID[3]].coords
            n4=nodes[ordered_ID[4]].coords
    
            points = [n1, n2, n3, n4]
    
            _x(n) = n[1]
            _y(n) = n[2]
            _z(n) = n[3]
    
            l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
            fn = unit_vector(l[2] × l[3]) # face normal vector
    
            angles=Float64[]
            theta2 = 0.0
            theta3 = angle1(l, 2, 3)*(signbit((l[2] × fn)⋅l[3]) ? 1 : -1)
            theta4 = angle1(l, 2, 4)*(signbit((l[2] × fn)⋅l[4]) ? 1 : -1)
    
            push!(angles,theta2,theta3,theta4)
    
            dict=Dict()
            for (n,f) in enumerate(angles)
                dict[f] = ordered_ID[n+1]
            end
    
            sorted_angles=sort(angles)
    
            sorted_IDs=Int64[]
            push!(sorted_IDs,ordered_ID[1])
            push!(sorted_IDs,dict[sorted_angles[1]])
            push!(sorted_IDs,dict[sorted_angles[2]])
            push!(sorted_IDs,dict[sorted_angles[3]])
    
            counter=0
            for i=iface_nodes_range[fID]
                counter=counter+1
                iface_nodes[i]=sorted_IDs[counter]
            end
        end
    end
    return bface_nodes, iface_nodes
end

function generate_cell_face_connectivity(cells_UNV, nbfaces, face_owner_cells)
    cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells_UNV)] 
    cell_nsign = Vector{Int64}[Int64[] for _ ∈ eachindex(cells_UNV)] 
    cell_neighbours = Vector{Int64}[Int64[] for _ ∈ eachindex(cells_UNV)] 
    cell_faces_range = UnitRange{Int64}[UnitRange{Int64}(0,0) for _ ∈ eachindex(cells_UNV)] 

    # Pass face ID to each cell
    first_internal_face = nbfaces + 1
    total_faces = length(face_owner_cells)
    for fID ∈ first_internal_face:total_faces
        owners = face_owner_cells[fID] # 2 cell owners IDs
        owner1 = owners[1]
        owner2 = owners[2]
        push!(cell_faces[owner1], fID) # Cell-faces only for internal faces.     
        push!(cell_faces[owner2], fID)
        push!(cell_nsign[owner1], 1) # Contract: Face normal goes from owner 1 to 2      
        push!(cell_nsign[owner2], -1)   
        push!(cell_neighbours[owner1], owner2)     
        push!(cell_neighbours[owner2], owner1)     
    end

    # Generate cell faces range
    start = 1
    for (cID, faces) ∈ enumerate(cell_faces)
        nfaces = length(faces)
        cell_faces_range[cID] = UnitRange{Int64}(start:(start + nfaces - 1))
        start += nfaces
    end

    # Flatten output (go from Vector{Vector{Int}} to Vector{Int})

    cell_faces = vcat(cell_faces...)
    cell_nsign = vcat(cell_nsign...)

    # NOTE: Check how this is being accessed in RANS models (need to flatten?)
    cell_neighbours = vcat(cell_neighbours...) # Need to check RANSMODELS!!!

    return cell_faces, cell_nsign, cell_faces_range, cell_neighbours
end

# CALCULATION of geometric properties for cells and faces

calculate_centres!(mesh) = begin
    (; nodes, cells, faces, cell_nodes, face_nodes) = mesh
    sum = SVector{3, Float64}(0.0,0.0,0.0)

    # calculate cell centres (geometric - not centroid - needs fixing)
    for cID ∈ eachindex(cells)
        cell = cells[cID]
        sum = SVector{3, Float64}(0.0,0.0,0.0)
        nodes_ID = nodeIDs(cell_nodes, cell.nodes_range)
        for nID ∈ nodes_ID
            sum += nodes[nID].coords 
        end
        @reset cell.centre = sum/length(nodes_ID)
        cells[cID] = cell
    end

    # calculate face centres (geometric - not centroid - needs fixing)
    for fID ∈ eachindex(faces)
        face = faces[fID]
        sum = SVector{3, Float64}(0.0,0.0,0.0)
        nodes_ID = nodeIDs(face_nodes, face.nodes_range)
        for nID ∈ nodes_ID
            sum += nodes[nID].coords 
        end
        @reset face.centre = sum/length(nodes_ID)
        faces[fID] = face
    end        
end

calculate_face_properties!(mesh) = begin
    # written for Tet only for debugging
    (; nodes, cells, faces, face_nodes, boundary_cellsID) = mesh
    n_bfaces = length(boundary_cellsID)
    n_faces = length(mesh.faces)

    # loop over boundary faces
    for fID ∈ 1:n_bfaces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        node1 = nodes[nIDs[1]]
        node2 = nodes[nIDs[2]]
        owners = face.ownerCells

        cell1 = cells[owners[1]]
        fc_n1 = node1.coords - face.centre
        fc_n2 = node2.coords - face.centre 
        cc1_cc2 = face.centre - cell1.centre

        normal_vec = fc_n1 × fc_n2
        normal = normal_vec/norm(normal_vec)
        if cc1_cc2 ⋅ normal < 0
            normal *= -1
        end
        @reset face.normal = normal

        # delta
        cc_fc = face.centre - cell1.centre
        delta = norm(cc_fc)
        e = cc_fc/delta
        weight = one(Float64)
        @reset face.delta = delta
        @reset face.e = e
        @reset face.weight = weight

        faces[fID] = face
    end

    # loop over internal faces
    for fID ∈ (n_bfaces + 1):n_faces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        node1 = nodes[nIDs[1]]
        node2 = nodes[nIDs[2]]
    
        owners = face.ownerCells
        cell1 = cells[owners[1]]
        cell2 = cells[owners[2]]
        fc_n1 = node1.coords - face.centre
        fc_n2 = node2.coords - face.centre 
        cc1_cc2 = cell2.centre - cell1.centre

        normal_vec = fc_n1 × fc_n2
        normal = normal_vec/norm(normal_vec)
        if cc1_cc2 ⋅ normal < 0
            normal *= -1
        end
        @reset face.normal = normal

        # delta
        c1_c2 = cell2.centre - cell1.centre
        fc_c1 = cell1.centre - face.centre
        fc_c2 = cell2.centre - face.centre
        # delta = norm(c1_c2)
        delta = norm(fc_c1) + norm(fc_c2)
        e = c1_c2/delta
        # weight = norm(fc_c2)/norm(c1_c2)
        weight = norm(fc_c2)/(norm(fc_c1) + norm(fc_c2))
        @reset face.delta = delta
        @reset face.e = e
        @reset face.weight = weight
        
        faces[fID] = face
    end
end

calculate_area_and_volume!(mesh) = begin
    (; nodes, faces, face_nodes, cells) = mesh

    n_faces=length(faces)

    #Using old method for now, know it works. Following method outlined by Sandip.

    #Calculating Area of Faces (Triangle and Quad)
    for fID ∈ 1:n_faces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        if length(face.nodes_range) == 3 # For Triangles
            
            n1 = nodes[nIDs[1]].coords
            n2 = nodes[nIDs[2]].coords
            n3 = nodes[nIDs[3]].coords
            
            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2
            
            @reset face.area = area

            faces[fID] = face
        end

        if length(face.nodes_range) == 4 # Quad Faces, Can be extended to faces with more nodes
            n1 = nodes[nIDs[1]].coords
            n2 = nodes[nIDs[2]].coords
            n3 = nodes[nIDs[3]].coords

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2

            for ic=4:4 # Temp fix for quad faces only.
                n1 = nodes[nIDs[1]].coords
                n2 = nodes[nIDs[3]].coords
                n3 = nodes[nIDs[ic]].coords # Make sure for a square that the opposite node is used so that it covers the entire face.

                t1x=n2[1]-n1[1]
                t1y=n2[2]-n1[2]
                t1z=n2[3]-n1[3]

                t2x=n3[1]-n1[1]
                t2y=n3[2]-n1[2]
                t2z=n3[3]-n1[3]

                area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
                area=area+sqrt(area2)/2

            end
            @reset face.area = area
            faces[fID] = face
        end
    end

    #Calculating all cell faces to calculate volume for each cell
    all_cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells)]
    for fID ∈ eachindex(faces)
        owners = faces[fID].ownerCells
        owner1 = owners[1]
        owner2 = owners[2]
        push!(all_cell_faces[owner1],fID)
        if owner1 !== owner2 #avoid duplication of cells for boundary faces
            push!(all_cell_faces[owner2],fID)
        end
    end

    #Calculating volume of each cell. Using Gaussian Divergence Theory method outlined by Sandip.
    for cID ∈ eachindex(cells)
        cell = cells[cID]
        nface = length(all_cell_faces[cID])
        volume = 0
        cc = cell.centre

        for f=1:nface
            ifc=all_cell_faces[cID][f]
            face = faces[ifc]

            normal=face.normal
            fc=face.centre
            d_fc=fc-cc

            if dot(d_fc,normal)<0.0
                normal=-1.0*normal
            end
            
            volume=volume+(normal[1]*fc[1]*face.area) #Only uses x direction. For better results, can be extended to y and z.

        end
        @reset cell.volume = volume
        cells[cID] = cell
    end
end