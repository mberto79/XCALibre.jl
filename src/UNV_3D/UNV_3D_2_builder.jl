export build_mesh3D
#push! is BANNED! THOSE FOUND USING PUSH WILL BE EXILED!
# [] is also BANNED! No more kneecaps if found using []!
#Exceptions apply to where I have no idea what the size will be

function build_mesh3D(unv_mesh; scale=1, integer=Int64, float=Float64)
    stats = @timed begin
        println("Loading UNV File...")
        points, edges, efaces, volumes, boundaryElements = load_3D(
            unv_mesh; scale=scale, integer=integer, float=float)
        println("File Read Successfully")
        println("Generating Mesh...")

        cell_nodes, cell_nodes_range = generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
        node_cells, node_cells_range = generate_node_cells(points, volumes)  # Should be Hybrid compatible, tested for hexa.
        nodes = build_nodes(points, node_cells_range) # Hyrbid compatible, works for Tet and Hexa
        boundaries = build_boundaries(boundaryElements) # Hybrid compatible

        nbfaces = sum(length.(getproperty.(boundaries, :IDs_range))) # total boundary faces

        bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = 
        begin
            generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes) # Hybrid compatible, tested with hexa
        end

        iface_nodes, iface_nodes_range, iface_owners_cells = 
        begin 
            generate_internal_faces(volumes, nbfaces, nodes, node_cells) # Hybrid compatible, tested with hexa.
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
            generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells) # Hybrid compatible. Hexa and tet tested.
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
        # return 0

#=
        face_nodes = generate_face_nodes(faces) #Removed push


        # all_cell_faces = generate_all_cell_faces(faces, cell_face_nodes) # New method needed
        # face_nodes_range = generate_face_nodes_range(faces) #Removed Push
        # all_cell_faces_range = generate_all_cell_faces_range(volumes) #Removed push

        cells_centre = calculate_centre_cell(volumes, nodes) #Removed push
        cell_faces, cell_faces_range = generate_cell_faces(bfaces, volumes, all_cell_faces) 
        face_ownerCells = generate_face_ownerCells(
            faces, all_cell_faces, all_cell_faces_range) #New method approach needed

        faces_area = calculate_face_area(nodes, faces) #Rewrite needed, removed push
        faces_centre = calculate_face_centre(faces, nodes) # Removed push
        faces_normal = calculate_face_normal(nodes, faces, face_ownerCells, cells_centre, faces_centre) # Rewrite needed
        faces_e, faces_delta, faces_weight = calculate_face_properties(faces, face_ownerCells, cells_centre, faces_centre, faces_normal) #Removed push

        cells_volume = calculate_cell_volume(volumes, all_cell_faces_range, all_cell_faces, faces_normal, cells_centre, faces_centre, face_ownerCells, faces_area) #Removed push

        cells = generate_cells(volumes, cells_centre, cells_volume, cell_nodes_range, cell_faces_range) #Removed push
        cell_neighbours = generate_cell_neighbours(cells, cell_faces) # Removed push, new method needed
        faces = generate_faces(faces, face_nodes_range, faces_centre, faces_normal, faces_area, face_ownerCells, faces_e, faces_delta, faces_weight) #Removed push

        cell_nsign = calculate_cell_nsign(cells, faces, cell_faces) #removed push

        get_float = SVector(0.0, 0.0, 0.0)
        get_int = UnitRange(0, 0)

        mesh = Mesh3(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes, boundaries, nodes, node_cells, get_float, get_int, boundary_cells)

    end
    println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    println("Mesh ready!")
    return mesh
    #For unit testing
    #return mesh,cell_face_nodes, node_cells, all_cell_faces,boundary_cells,boundary_faces,all_cell_faces_range
=#
    end
end

# function order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes,nodes) # New Method
#     n_bfaces = length(bface_nodes_range)
#     n_ifaces =  length(iface_nodes_range)
#     for fID = 1:n_bfaces # Re-order Boundary Faces
#         if length(bface_nodes_range[fID]) == 4 # Only for Quad faces
#             nIDs=nodeIDs(bface_nodes,bface_nodes_range[fID]) # Get ids of nodes of face
    
#             ordered_ID=sort(nIDs) # sort them so that the lowest ID is first
    
#             p1=nodes[ordered_ID[1]].coords # Get coords of 4 nodes
#             p2=nodes[ordered_ID[2]].coords
#             p3=nodes[ordered_ID[3]].coords
#             p4=nodes[ordered_ID[4]].coords
    
#             points = [p1, p2, p3, p4]
    
#             _x(p) = p[1]
#             _y(p) = p[2]
#             _z(p) = p[3]

#             p1_p2 = segment(p1, p2) # vector/segment from p1 to p2 (pivot!)
#             p1_p3 = segment(p1, p3) # vector/segment from p1 to p3
#             p1_p4 = segment(p1, p4) # vector/segment from p1 to p4

#             angle2_2 = 0.0
#             angle2_3 = angle1(p1_p2, p1_p3)
#             angle2_4 = angle1(p1_p2, p1_p4)
    
#             angles=Float64[]
#             push!(angles,angle2_2,angle2_3,angle2_4)
    
#             dict=Dict() # Using dictionary to link noode Id to angle
#             for (n,f) in enumerate(angles)
#                 dict[f] = ordered_ID[n+1]
#             end
    
#             sorted_angles=sort(angles) # Sort angles from smallest to largest. Right hand rule.
    
#             sorted_IDs=Int64[]
#             push!(sorted_IDs,ordered_ID[1])
#             push!(sorted_IDs,dict[sorted_angles[1]])
#             push!(sorted_IDs,dict[sorted_angles[2]])
#             push!(sorted_IDs,dict[sorted_angles[3]])
    
#             counter=0
#             for i=bface_nodes_range[fID] # Re-writing face_nodes with ordered nodes
#                 counter=counter+1
#                 bface_nodes[i]=sorted_IDs[counter]
#             end
#         end
#     end
    
#     for fID = 1:n_ifaces # Re-order internal faces
#         if length(iface_nodes_range[fID])==4 # Only for Quad Faces
#             nIDs=nodeIDs(iface_nodes,iface_nodes_range[fID]) # Get ids of nodes of the face
    
#             ordered_ID=sort(nIDs) # Sort them so that the lowest ID is first
    
#             p1=nodes[ordered_ID[1]].coords # Get coords of 4 nodes
#             p2=nodes[ordered_ID[2]].coords
#             p3=nodes[ordered_ID[3]].coords
#             p4=nodes[ordered_ID[4]].coords
    
#             points = [p1, p2, p3, p4]
    
#             _x(p) = p[1]
#             _y(p) = p[2]
#             _z(p) = p[3]
    
#             p1_p2 = segment(p1, p2) # vector/segment from p1 to p2 (pivot!)
#             p1_p3 = segment(p1, p3) # vector/segment from p1 to p3
#             p1_p4 = segment(p1, p4) # vector/segment from p1 to p4

#             angle2_2 = 0.0
#             angle2_3 = angle1(p1_p2, p1_p3)
#             angle2_4 = angle1(p1_p2, p1_p4)
    
#             angles=Float64[]
#             push!(angles,angle2_2,angle2_3,angle2_4)
    
#             dict=Dict()
#             for (n,f) in enumerate(angles)
#                 dict[f] = ordered_ID[n+1]
#             end
    
#             sorted_angles=sort(angles)
    
#             sorted_IDs=Int64[]
#             push!(sorted_IDs,ordered_ID[1])
#             push!(sorted_IDs,dict[sorted_angles[1]])
#             push!(sorted_IDs,dict[sorted_angles[2]])
#             push!(sorted_IDs,dict[sorted_angles[3]])
    
#             counter=0
#             for i=iface_nodes_range[fID]
#                 counter=counter+1
#                 iface_nodes[i]=sorted_IDs[counter]
#             end
#         end
#     end
#     return bface_nodes, iface_nodes
# end

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
    
            fc=sum(points)/length(points) # geographic centre (not centroid)
            s = segment.(Ref(fc), points) # surface vectors (from face centre)
            u = unit_vector.(s)
            l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
            fn = unit_vector(l[2] × l[3]) # face normal vector
    
            angles=Float64[] # Vector to store angles
            #theta2 = angle1(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
            theta2 = 0.0
            theta3 = angle1(l, 2, 3)*(signbit((l[2] × fn)⋅l[3]) ? 1 : -1)
            theta4 = angle1(l, 2, 4)*(signbit((l[2] × fn)⋅l[4]) ? 1 : -1)
    
            push!(angles,theta2,theta3,theta4)
    
            dict=Dict() # Using dictionary to link noode Id to angle
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
    
            fc=sum(points)/length(points) # geographic centre (not centroid)
            s = segment.(Ref(fc), points) # surface vectors (from face centre)
            u = unit_vector.(s)
            l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
            fn = unit_vector(l[2] × l[3]) # face normal vector
    
            angles=Float64[]
            #theta2 = angle1(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
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

build_mesh() = begin
    nothing
    # return mesh
end

# CALCULATION of mesh properties

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
        #node3 = nodes[nIDs[3]]
        # edge1 = node2.coords - node1.coords
        # edge2 = node3.coords - node1.coords
        owners = face.ownerCells

        cell1 = cells[owners[1]]
        # cell2 = cells[owners[2]]
        fc_n1 = node1.coords - face.centre
        fc_n2 = node2.coords - face.centre 
        # cc1_cc2 = cell2.centre - cell1.centre
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
        #@reset face.area = 0.5*norm(edge1×edge2)

        faces[fID] = face
    end

    # loop over internal faces
    for fID ∈ (n_bfaces + 1):n_faces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        node1 = nodes[nIDs[1]]
        node2 = nodes[nIDs[2]]
        #node3 = nodes[nIDs[3]]
        # edge1 = node2.coords - node1.coords
        # edge2 = node3.coords - node1.coords
        owners = face.ownerCells
        cell1 = cells[owners[1]]
        cell2 = cells[owners[2]]
        fc_n1 = node1.coords - face.centre
        fc_n2 = node2.coords - face.centre 
        cc1_cc2 = cell2.centre - cell1.centre
        # cc1_cc2 = face.centre - cell1.centre
        normal_vec = fc_n1 × fc_n2
        normal = normal_vec/norm(normal_vec)
        if cc1_cc2 ⋅ normal < 0
            normal *= -1
        end
        @reset face.normal = normal

        # delta
        c1_c2 = cell2.centre - cell1.centre
        fc_c1 = face.centre - cell1.centre
        c2_fc = cell2.centre - face.centre
        delta = norm(c1_c2)
        e = c1_c2/delta
        weight = abs((fc_c1 ⋅ normal)/((fc_c1 ⋅ normal)+(c2_fc ⋅ normal)))
        @reset face.delta = delta
        @reset face.e = e
        @reset face.weight = weight
        #@reset face.area = 0.5*norm(edge1×edge2)
        
        faces[fID] = face
    end
end

# calculate_face_properties!(mesh) = begin
#     # written for Tet only for debugging
#     (; nodes, cells, faces, face_nodes, boundary_cellsID) = mesh
#     n_bfaces = length(boundary_cellsID)
#     n_faces = length(mesh.faces)

#     # loop over boundary faces
#     for fID ∈ 1:n_bfaces
#         face = faces[fID]
#         nIDs = nodeIDs(face_nodes, face.nodes_range)
#         node1 = nodes[nIDs[1]]
#         node2 = nodes[nIDs[2]]
#         node3 = nodes[nIDs[3]]
#         edge1 = node2.coords - node1.coords
#         edge2 = node3.coords - node1.coords
#         owners = face.ownerCells

#         cell1 = cells[owners[1]]
#         # cell2 = cells[owners[2]]
#         fc_n1 = node1.coords - face.centre
#         fc_n2 = node2.coords - face.centre 
#         # cc1_cc2 = cell2.centre - cell1.centre
#         cc1_cc2 = face.centre - cell1.centre
#         normal_vec = fc_n1 × fc_n2
#         normal = normal_vec/norm(normal_vec)
#         if cc1_cc2 ⋅ normal < 0
#             normal *= -1
#         end
#         @reset face.normal = normal

#         # delta
#         cc_fc = face.centre - cell1.centre
#         delta = norm(cc_fc)
#         e = cc_fc/delta
#         weight = one(Float64)
#         @reset face.delta = delta
#         @reset face.e = e
#         @reset face.weight = weight
#         @reset face.area = 0.5*norm(edge1×edge2)

#         faces[fID] = face
#     end

#     # loop over internal faces
#     for fID ∈ (n_bfaces + 1):n_faces
#         face = faces[fID]
#         nIDs = nodeIDs(face_nodes, face.nodes_range)
#         node1 = nodes[nIDs[1]]
#         node2 = nodes[nIDs[2]]
#         node3 = nodes[nIDs[3]]
#         edge1 = node2.coords - node1.coords
#         edge2 = node3.coords - node1.coords
#         owners = face.ownerCells
#         cell1 = cells[owners[1]]
#         cell2 = cells[owners[2]]
#         fc_n1 = node1.coords - face.centre
#         fc_n2 = node2.coords - face.centre 
#         cc1_cc2 = cell2.centre - cell1.centre
#         # cc1_cc2 = face.centre - cell1.centre
#         normal_vec = fc_n1 × fc_n2
#         normal = normal_vec/norm(normal_vec)
#         if cc1_cc2 ⋅ normal < 0
#             normal *= -1
#         end
#         @reset face.normal = normal

#         # delta
#         c1_c2 = cell2.centre - cell1.centre
#         delta = norm(c1_c2)
#         e = c1_c2/delta
#         weight = 0.5 # !!!!!!!!!!!!!! fixed value for now !!!!!!!!!!
#         @reset face.delta = delta
#         @reset face.e = e
#         @reset face.weight = weight
#         @reset face.area = 0.5*norm(edge1×edge2)
        
#         faces[fID] = face
#     end
# end

calculate_area_and_volume!(mesh) = begin
    (; nodes, faces, face_nodes, cells, cell_nodes) = mesh

    n_faces=length(faces)

    #Using old method for now, know it works. Following method outlined by Sandip.

    for fID ∈ 1:n_faces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        if length(face.nodes_range) == 3 # For Triangles
            #nIDs = nodeIDs(face_nodes, face.nodes_range)
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

        if length(face.nodes_range) == 4 # Quad Faces Can be extended
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

            for ic=4:4 # Temp fix
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

    all_cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells)]
    for fID ∈ eachindex(faces)
        owners = faces[fID].ownerCells
        owner1 = owners[1]
        owner2 = owners[2]
        #if faces_cpu[fID].ownerCells[1]==cID || faces_cpu[fID].ownerCells[2]==cID
            push!(all_cell_faces[owner1],fID)
            if owner1 !== owner2 #avoid duplication of cells for boundary faces
                push!(all_cell_faces[owner2],fID)
            end
        #end
    end

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

            #if  face.ownerCells[1] ≠ face.ownerCells[2]
                if dot(d_fc,normal)<0.0
                    normal=-1.0*normal
                end
            #end

            volume=volume+(normal[1]*fc[1]*face.area)

        end
        @reset cell.volume = volume
        cells[cID] = cell
    end
end

# calculate_area_and_volume!(mesh) = begin
#     (; nodes, faces, face_nodes, cells, cell_nodes) = mesh

#     n_faces=length(faces)

#     #Using old method for now, know it works. Following method outlined by Sandip.

#     for fID ∈ 1:n_faces
#         face = faces[fID]
#         nIDs = nodeIDs(face_nodes, face.nodes_range)
#         if length(face.nodes_range) == 3 # For Triangles
#             #nIDs = nodeIDs(face_nodes, face.nodes_range)
#             n1 = nodes[nIDs[1]].coords
#             n2 = nodes[nIDs[2]].coords
#             n3 = nodes[nIDs[3]].coords
            
#             t1x=n2[1]-n1[1]
#             t1y=n2[2]-n1[2]
#             t1z=n2[3]-n1[3]

#             t2x=n3[1]-n1[1]
#             t2y=n3[2]-n1[2]
#             t2z=n3[3]-n1[3]

#             area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
#             area=sqrt(area2)/2
            
#             @reset face.area = area

#             faces[fID] = face
#         end

#         if length(face.nodes_range) >= 4 # For any shape
#             n1 = nodes[nIDs[1]].coords
#             n2 = nodes[nIDs[2]].coords
#             n3 = nodes[nIDs[3]].coords

#             t1x=n2[1]-n1[1]
#             t1y=n2[2]-n1[2]
#             t1z=n2[3]-n1[3]

#             t2x=n3[1]-n1[1]
#             t2y=n3[2]-n1[2]
#             t2z=n3[3]-n1[3]

#             area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
#             area=sqrt(area2)/2

#             for ic=4:length(face.nodes_range)
#                 n1 = nodes[nIDs[ic]].coords
#                 n2 = nodes[nIDs[2]].coords
#                 n3 = nodes[nIDs[3]].coords

#                 t1x=n2[1]-n1[1]
#                 t1y=n2[2]-n1[2]
#                 t1z=n2[3]-n1[3]

#                 t2x=n3[1]-n1[1]
#                 t2y=n3[2]-n1[2]
#                 t2z=n3[3]-n1[3]

#                 area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
#                 area=area+sqrt(area2)/2

#             end
#             @reset face.area = area
#             faces[fID] = face
#         end
#     end


#     c = 1/6 # calculate only once
#     for cID ∈ eachindex(cells)
#         cell = cells[cID]
#         # fIDs = faceIDs(cell_faces, cell.faces_range)
#         # face = faces[fIDs[1]]
#         # nIDs = nodeIDs(face_nodes, face.nodes_range)
#         nIDs = nodeIDs(cell_nodes, cell.nodes_range)
#         node1 = nodes[nIDs[1]]
#         node2 = nodes[nIDs[2]]
#         node3 = nodes[nIDs[3]]
#         node4 = nodes[nIDs[4]]
#         edge1 = node2.coords - node1.coords
#         edge2 = node3.coords - node1.coords
#         edge3 = node4.coords - node1.coords
#         # dist = norm(node2.coords - node1.coords)

#         volume = c*((edge1×edge2)⋅edge3)
#         @reset cell.volume = volume
#         cells[cID] = cell
#     end
# end

# calculate_area_and_volume!(mesh) = begin
#     (; nodes, faces, face_nodes, cells, cell_faces, cell_nodes) = mesh
#     c = 1/6 # calculate only once
#     for cID ∈ eachindex(cells)
#         cell = cells[cID]
#         # fIDs = faceIDs(cell_faces, cell.faces_range)
#         # face = faces[fIDs[1]]
#         # nIDs = nodeIDs(face_nodes, face.nodes_range)
#         nIDs = nodeIDs(cell_nodes, cell.nodes_range)
#         node1 = nodes[nIDs[1]]
#         node2 = nodes[nIDs[2]]
#         node3 = nodes[nIDs[3]]
#         node4 = nodes[nIDs[4]]
#         edge1 = node2.coords - node1.coords
#         edge2 = node3.coords - node1.coords
#         edge3 = node4.coords - node1.coords
#         # dist = norm(node2.coords - node1.coords)

#         volume = c*((edge1×edge2)⋅edge3)
#         @reset cell.volume = volume
#         cells[cID] = cell
#     end
# end

# Node connectivity

function generate_node_cells(points, volumes)
    temp_node_cells = [Int64[] for _ ∈ eachindex(points)] # array of vectors to hold cellIDs

    # Add cellID to each point that defines a "volume"
    for (cellID, volume) ∈ enumerate(volumes)
        for nodeID ∈ volume.volumes
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

# function generate_node_cells(points, volumes)
#     temp_node_cells = [Int64[] for _ ∈ eachindex(points)] # array of vectors to hold cellIDs

#     # Add cellID to each point that defines a "volume"
#     for (cellID, volume) ∈ enumerate(volumes)
#         for nodeID ∈ volume.volumes
#             push!(temp_node_cells[nodeID], cellID)
#         end
#     end

#     node_cells_size = sum(length.(temp_node_cells)) # number of cells in node_cells

#     index = 0 # change to node cells index
#     node_cells = zeros(Int64, node_cells_size)
#     node_cells_range = [UnitRange{Int64}(1, 1) for _ ∈ eachindex(points)]
#     for (nodeID, cellsID) ∈ enumerate(temp_node_cells)
#         for cellID ∈ cellsID
#             index += 1
#             node_cells[index] = cellID
#         end
#         node_cells_range[nodeID] = UnitRange{Int64}(index - length(cellsID) + 1, index)
#     end
#     return node_cells, node_cells_range
# end

function build_nodes(points, node_cells_range) # Hybrid compatible. Works for Tet and Hexa
    nodes = [Node(SVector{3, Float64}(0.0,0.0,0.0), 1:1) for _ ∈ eachindex(points)]
    @inbounds for i ∈ eachindex(points)
        nodes[i] =  Node(points[i].xyz, node_cells_range[i])
    end
    return nodes
end

# function build_nodes(points, cells_range)
#     nodes = [Node(SVector{3, Float64}(0.0,0.0,0.0), 1:1) for _ ∈ eachindex(points)]
#     @inbounds for i ∈ eachindex(points)
#         nodes[i] =  Node(points[i].xyz, cells_range[i])
#     end
#     return nodes
# end

# DEFINE FUNCTIONS
function calculate_face_properties(faces, face_ownerCells, cells_centre, faces_centre, face_normal)
    faces_e = Vector{SVector{3,Float64}}(undef,length(faces))
    faces_delta = Vector{Float64}(undef,length(faces))
    faces_weight = Vector{Float64}(undef,length(faces))
    for i = eachindex(faces) #Boundary Face
        if face_ownerCells[i, 2] == face_ownerCells[i, 1]
            cc = cells_centre[face_ownerCells[i, 1]]
            cf = faces_centre[i]

            d_cf = cf - cc

            delta = norm(d_cf)
            faces_delta[i] = delta
            e = d_cf / delta
            faces_e[i] = e
            weight = one(Float64)
            faces_weight[i] = weight

        else #Internal Face
            c1 = cells_centre[face_ownerCells[i, 1]]
            c2 = cells_centre[face_ownerCells[i, 2]]
            cf = faces_centre[i]
            d_1f = cf - c1
            d_f2 = c2 - cf
            d_12 = c2 - c1

            delta = norm(d_12)
            faces_delta[i] = delta
            e = d_12 / delta
            faces_e[i] = e
            weight = abs((d_1f ⋅ face_normal[i]) / (d_1f ⋅ face_normal[i] + d_f2 ⋅ face_normal[i]))
            faces_weight[i] = weight

        end
    end
    return faces_e, faces_delta, faces_weight
end

function calculate_face_normal(nodes, faces, face_ownerCells, cells_centre, faces_centre) #Rewrite needed
    face_normal = Vector{SVector{3,Float64}}(undef,length(faces))
    for i = eachindex(faces)
        n1 = nodes[faces[i].faces[1]].coords
        n2 = nodes[faces[i].faces[2]].coords
        n3 = nodes[faces[i].faces[3]].coords

        t1x = n2[1] - n1[1]
        t1y = n2[2] - n1[2]
        t1z = n2[3] - n1[3]

        t2x = n3[1] - n1[1]
        t2y = n3[2] - n1[2]
        t2z = n3[3] - n1[3]

        nx = t1y * t2z - t1z * t2y
        ny = -(t1x * t2z - t1z * t2x)
        nz = t1x * t2y - t1y * t2x

        magn2 = (nx)^2 + (ny)^2 + (nz)^2

        snx = nx / sqrt(magn2)
        sny = ny / sqrt(magn2)
        snz = nz / sqrt(magn2)

        normal = SVector(snx, sny, snz)
        face_normal[i] = normal

        if face_ownerCells[i, 2] == face_ownerCells[i, 1]
            cc = cells_centre[face_ownerCells[i, 1]]
            cf = faces_centre[i]

            d_cf = cf - cc

            if d_cf ⋅ face_normal[i] < 0
                face_normal[i] = -1.0 * face_normal[i]
            end
        else
            c1 = cells_centre[face_ownerCells[i, 1]]
            c2 = cells_centre[face_ownerCells[i, 2]]
            cf = faces_centre[i]
            d_12 = c2 - c1

            if d_12 ⋅ face_normal[i] < 0
                face_normal[i] = -1.0 * face_normal[i]
            end
        end
    end
    return face_normal
end

function calculate_face_centre(faces, nodes)
    face_centres = Vector{SVector{3,Float64}}(undef,length(faces))
    for i = eachindex(faces)
        temp_coords = Vector{SVector{3,Float64}}(undef,length(faces[i].faces))
        for ic = 1:length(faces[i].faces)
            temp_coords[ic] = nodes[faces[i].faces[ic]].coords
        end
        face_centres[i] = sum(temp_coords) / length(faces[i].faces)
    end
    return face_centres
end

function calculate_face_area(nodes, faces) # Need to shorten
    face_area= Vector{Float64}(undef,length(faces))
    for i = eachindex(faces)
        if faces[i].faceCount == 3
            n1 = nodes[faces[i].faces[1]].coords
            n2 = nodes[faces[i].faces[2]].coords
            n3 = nodes[faces[i].faces[3]].coords

            t1x = n2[1] - n1[1]
            t1y = n2[2] - n1[2]
            t1z = n2[3] - n1[3]

            t2x = n3[1] - n1[1]
            t2y = n3[2] - n1[2]
            t2z = n3[3] - n1[3]

            area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
            area = sqrt(area2) / 2
            face_area[i]= area
        end

        if faces[i].faceCount > 3
            n1 = nodes[faces[i].faces[1]].coords
            n2 = nodes[faces[i].faces[2]].coords
            n3 = nodes[faces[i].faces[3]].coords

            t1x = n2[1] - n1[1]
            t1y = n2[2] - n1[2]
            t1z = n2[3] - n1[3]

            t2x = n3[1] - n1[1]
            t2y = n3[2] - n1[2]
            t2z = n3[3] - n1[3]

            area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
            area = sqrt(area2) / 2

            for ic = 4:faces[i].faceCount
                n1 = nodes[faces[i].faces[ic]].coords
                n2 = nodes[faces[i].faces[2]].coords
                n3 = nodes[faces[i].faces[3]].coords

                t1x = n2[1] - n1[1]
                t1y = n2[2] - n1[2]
                t1z = n2[3] - n1[3]

                t2x = n3[1] - n1[1]
                t2y = n3[2] - n1[2]
                t2z = n3[3] - n1[3]

                area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
                area = area + sqrt(area2) / 2

            end

            face_area[i]=area

        end
    end
    return face_area
end

function generate_cell_faces(bfaces, volumes, all_cell_faces)
    cell_faces = Vector{Int64}[] # May be a way to preallocate this. For now will leave as []
    cell_face_range = Vector{UnitRange{Int64}}(undef,length(volumes))
    counter_start = 0
    x = 0
    max = length(bfaces)

    for i = eachindex(volumes)
        push!(cell_faces, all_cell_faces[counter_start+1:counter_start+length(volumes[i].volumes)])
        counter_start = counter_start + length(volumes[i].volumes)
        cell_faces[i] = filter(x -> x > max, cell_faces[i])

        if length(cell_faces[i]) == 1
            cell_face_range[i] = UnitRange(x + 1, x + 1)
            x = x + 1
        else
            cell_face_range[i] = UnitRange(x + 1, x + length(cell_faces[i]))
            x = x + length(cell_faces[i])
        end
    end
    cell_faces = reduce(vcat, cell_faces)

    return cell_faces, cell_face_range
end

function calculate_cell_nsign(cells, faces, cell_faces)
    cell_nsign = Vector{Int}(undef,length(cell_faces))
    counter=0
    for i = 1:length(cells)
        centre = cells[i].centre
        for ic = 1:length(cells[i].faces_range)
            fcentre = faces[cell_faces[cells[i].faces_range][ic]].centre
            fnormal = faces[cell_faces[cells[i].faces_range][ic]].normal
            d_cf = fcentre - centre
            fnsign = zero(Int)

            if d_cf ⋅ fnormal > zero(Float64)
                fnsign = one(Int)
            else
                fnsign = -one(Int)
            end
            counter=counter+1
            cell_nsign[counter] = fnsign
        end

    end
    return cell_nsign
end

function calculate_cell_volume(volumes, all_cell_faces_range, all_cell_faces, face_normal, cells_centre, faces_centre, face_ownerCells, faces_area)
    cells_volume = Vector{Float64}(undef,length(volumes))
    for i = eachindex(volumes)
        volume = zero(Float64) # to avoid type instability
        for f = all_cell_faces_range[i]
            findex = all_cell_faces[f]

            normal = face_normal[findex]
            cc = cells_centre[i]
            fc = faces_centre[findex]
            d_fc = fc - cc

            if face_ownerCells[findex, 1] ≠ face_ownerCells[findex, 2]
                if dot(d_fc, normal) < 0.0
                    normal = -1.0 * normal
                end
            end

            volume = volume + (normal[1] * faces_centre[findex][1] * faces_area[findex])

        end
        cells_volume[i] = volume
    end
    return cells_volume
end

function calculate_centre_cell(volumes,nodes)
    cell_centres = Vector{SVector{3,Float64}}(undef,length(volumes))
    for i = eachindex(volumes)
        temp_coords = Vector{SVector{3,Float64}}(undef,length(volumes[i].volumes))
        for ic = eachindex(volumes[i].volumes)
            temp_coords[ic] = nodes[volumes[i].volumes[ic]].coords
        end
        cell_centres[i] = sum(temp_coords) / length(volumes[i].volumes)
    end
    return cell_centres
end

# function calculate_centre_cell(volumes, nodes)
#     centre_store = SVector{3,Float64}[]
#     for i = 1:length(volumes)
#         cell_store = typeof(nodes[volumes[1].volumes[1]].coords)[]
#         for ic = 1:length(volumes[i].volumes)
#             push!(cell_store, nodes[volumes[i].volumes[ic]].coords)
#         end
#         centre = (sum(cell_store) / length(cell_store))
#         push!(centre_store, centre)
#     end
#     return centre_store
# end

function generate_cell_neighbours(cells, cell_faces)
    cell_neighbours = Vector{Int64}(undef,length(cell_faces))
    counter=0
    for ID = 1:length(cells)
        for i = cells[ID].faces_range
            faces = cell_faces[i]
            for ic = 1:length(i)
                face = faces[ic]
                index = findall(x -> x == face, cell_faces)
                if length(index) == 2
                    if i[1] <= index[1] <= i[end]
                        for ip = 1:length(cells)
                            if cells[ip].faces_range[1] <= index[2] <= cells[ip].faces_range[end]
                                counter=counter+1
                                cell_neighbours[counter] = ip
                            end
                        end
                    end
                    if i[1] <= index[2] <= i[end]
                        for ip = 1:length(cells)
                            if cells[ip].faces_range[1] <= index[1] <= cells[ip].faces_range[end]
                                counter=counter+1
                                cell_neighbours[counter] = ip
                            end
                        end
                    end
                end
            end
        end
    end
    return cell_neighbours
end

# NOTE: the function has been written to be extendable to multiple element types

function generate_internal_faces(volumes, nbfaces, nodes, node_cells)
    # determine total number of faces based on cell type (including duplicates)
    total_faces = 0
    for volume ∈ volumes
        # add faces for tets
        if volume.volumeCount == 4
            total_faces += 4
        end
        # add conditions to add faces for other cell types
        # add faces for Hexa
        if volume.volumeCount == 8 
            total_faces += 6
        end
        #add faces for Wedge/Penta
        if volume.volumeCount == 6
            total_faces += 5
        end
    end

    # Face nodeIDs for each cell is a vector of vectors of vectors :-)
    cells_faces_nodeIDs = Vector{Vector{Int64}}[Vector{Int64}[] for _ ∈ 1:length(volumes)] 

    # Generate all faces for each cell/element/volume
    for (cellID, volume) ∈ enumerate(volumes)
        # Generate faces for tet elements
        if volume.volumeCount == 4
            nodesID = volume.volumes
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[4]])
        end
        # add conditions for other cell types
        # Generate faces for hexa elements using UNV structure method
        if volume.volumeCount == 8
            nodesID = volume.volumes
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[5], nodesID[6], nodesID[7], nodesID[8]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[5], nodesID[6]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[3], nodesID[4], nodesID[7], nodesID[8]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[6], nodesID[7]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[4], nodesID[5], nodesID[8]])
        end
        # Generate faces for prism elements, using pattern in UNV file.
        if volume.volumeCount == 6
            nodesID = volume.volumes
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]]) # Triangle 1
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[3], nodesID[4], nodesID[5]]) # Triangle 2
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

function generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells)
    cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(volumes)] 
    cell_nsign = Vector{Int64}[Int64[] for _ ∈ eachindex(volumes)] 
    cell_neighbours = Vector{Int64}[Int64[] for _ ∈ eachindex(volumes)] 
    cell_faces_range = UnitRange{Int64}[UnitRange{Int64}(0,0) for _ ∈ eachindex(volumes)] 

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


function build_boundaries(boundaryElements)
    bfaces_start = 1
    boundaries = Vector{Boundary{Symbol,UnitRange{Int64}}}(undef,length(boundaryElements))
    for (i, boundaryElement) ∈ enumerate(boundaryElements)
        bfaces = length(boundaryElement.elements)
        bfaces_range = UnitRange{Int64}(bfaces_start:(bfaces_start + bfaces - 1))
        boundaries[i] = Boundary(Symbol(boundaryElement.name), bfaces_range)
        bfaces_start += bfaces
    end
    return boundaries
end

function generate_boundary_cells(bfaces, all_cell_faces, all_cell_faces_range)
    boundary_cells = Vector{Int64}(undef,length(bfaces))
    index_all_cell_faces = Vector{Int64}(undef,length(bfaces))
    for ic = eachindex(bfaces) 
        for i in eachindex(all_cell_faces) 
                if all_cell_faces[i] == bfaces[ic].faceindex
                    index_all_cell_faces[ic]=i
                end
        end
        for i = eachindex(all_cell_faces_range) 
            if all_cell_faces_range[i][1] <= index_all_cell_faces[ic] <= all_cell_faces_range[i][end]
                boundary_cells[ic]=i
            end
        end
    end
    return boundary_cells
end

function generate_boundary_faces(
    boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes
    )
    bface_nodes = Vector{Vector{Int64}}(undef, nbfaces)
    bface_nodes_range = Vector{UnitRange{Int64}}(undef, nbfaces)
    bowners_cells = Vector{Int64}[Int64[0,0] for _ ∈ 1:nbfaces]
    boundary_cells = Vector{Int64}(undef, nbfaces)

    fID = 0 # faceID index of output array (reordered)
    start = 1
    for boundary ∈ boundaryElements
        elements = boundary.elements
            for bfaceID ∈ elements
                fID += 1
                nnodes = length(efaces[bfaceID].faces)
                nodeIDs = efaces[bfaceID].faces # Actually nodesIDs
                bface_nodes[fID] = nodeIDs
                bface_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
                start += nnodes

                # Find owner cells (same as boundary cells)
                assigned = false
                for nodeID ∈ nodeIDs
                    cIDs = cellIDs(node_cells, node_cells_range, nodeID)
                    for cID ∈ cIDs
                        if intersect(nodeIDs, volumes[cID].volumes) == nodeIDs
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

# function generate_boundary_faces(
#     boundaryElements, bfaces, node_cells, node_cells_range, volumes
#     )
#     bface_nodes = Vector{Vector{Int64}}(undef, length(bfaces))
#     bface_nodes_range = Vector{UnitRange{Int64}}(undef,length(bfaces))
#     bowners_cells = Vector{Int64}[Int64[0,0] for _ ∈ eachindex(bfaces)]
#     boundary_cells = Vector{Int64}(undef,length(bfaces))

#     fID = 0 # faceID index of output array (reordered)
#     start = 1
#     for boundary ∈ boundaryElements
#         elements = boundary.elements
#             for bfaceID ∈ elements
#                 fID += 1
#                 nnodes = length(bfaces[bfaceID].faces)
#                 nodeIDs = bfaces[bfaceID].faces # Actually nodesIDs
#                 bface_nodes[fID] = nodeIDs
#                 bface_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
#                 start += nnodes

#                 # Find owner cells (same as boundary cells)
#                 assigned = false
#                 for nodeID ∈ nodeIDs
#                     cIDs = cellIDs(node_cells, node_cells_range, nodeID)
#                     for cID ∈ cIDs
#                         if intersect(nodeIDs, volumes[cID].volumes) == nodeIDs
#                             bowners_cells[fID] .= cID
#                             boundary_cells[fID] = cID
#                             assigned = true
#                             break
#                         end
#                     end
#                     if assigned 
#                         break
#                     end
#                 end
#             end
#     end

#     bface_nodes = vcat(bface_nodes...) # Flatten - from Vector{Vector{Int}} to Vector{Int}

#     return bface_nodes, bface_nodes_range, bowners_cells, boundary_cells
#     # return bfaces_nodes, bowners_cells
# end

#Generate Faces

function generate_face_nodes(faces)
    face_nodes = Vector{Int64}(undef, length(faces) * 3) # number of bc faces times number of nodes per face. Tet Only for now.
    counter = 0
    for n = eachindex(faces)
        for i = 1:faces[n].faceCount
            counter = counter + 1
            face_nodes[counter] = faces[n].faces[i]
        end
    end
    return face_nodes
end

#Generate cells

function generate_cell_nodes(volumes)
    #cell_nodes = Vector{Int64}(undef, length(volumes) * 4) #length of cells times number of nodes per cell (tet only)
    cell_nodes = Int64[] # cell_node length is undetermined as mesh could be hybrid, using push. Could use for and if before to preallocate vector.
    for n = eachindex(volumes)
        for i = 1:volumes[n].volumeCount
            push!(cell_nodes,volumes[n].volumes[i])
        end
    end

    cell_nodes_range = Vector{UnitRange{Int64}}(undef, length(volumes)) # cell_nodes_range determined by no. of cells.
    x = 0
    for i = eachindex(volumes)
        cell_nodes_range[i] = UnitRange(x + 1, x + length(volumes[i].volumes))
        x = x + length(volumes[i].volumes)
    end
    return cell_nodes, cell_nodes_range
end

# function generate_cell_nodes(volumes)
#     cell_nodes = Vector{Int64}(undef, length(volumes) * 4) #length of cells times number of nodes per cell (tet only)
#     counter = 0
#     for n = eachindex(volumes)
#         for i = 1:volumes[n].volumeCount
#             counter = counter + 1
#             cell_nodes[counter] = volumes[n].volumes[i]
#         end
#     end

#     cell_nodes_range = Vector{UnitRange{Int64}}(undef, length(volumes))
#     x = 0
#     for i = eachindex(volumes)
#         cell_nodes_range[i] = UnitRange(x + 1, x + length(volumes[i].volumes))
#         x = x + length(volumes[i].volumes)
#     end
#     return cell_nodes, cell_nodes_range
# end

function generate_all_cell_faces(faces, cell_face_nodes)
    sorted_faces = Vector{Vector{Int64}}(undef, length(faces))
    for i = 1:length(faces)
        sorted_faces[i] = sort(faces[i].faces)
    end

    all_cell_faces = zeros(Int, length(cell_face_nodes)) #May only work for Tet
    for i = 1:length(cell_face_nodes)
        all_cell_faces[i] = findfirst(x -> x == cell_face_nodes[i], sorted_faces)
    end
    return all_cell_faces
end


function generate_face_nodes_range(faces)
    face_nodes_range = Vector{UnitRange{Int64}}(undef, length(faces))
    x = 0
    for i = eachindex(faces)
        face_nodes_range[i] = UnitRange(x + 1, x + faces[i].faceCount)
        x = x + faces[i].faceCount
    end
    return face_nodes_range
end

function generate_all_cell_faces_range(volumes)
    cell_faces_range = Vector{UnitRange{Int64}}(undef, length(volumes))
    x = 0
    @inbounds for i = eachindex(volumes)
        #Tetra
        if length(volumes[i].volumes) == 4
            cell_faces_range[i] = UnitRange(x + 1, x + 4)
            x = x + 4
        end

        #Hexa
        if length(volumes[i].volumes) == 8
            cell_faces_range[i] = UnitRange(x + 1, x + 6)
            x = x + 6
        end
    end
    return cell_faces_range
end

#Generate cells
function generate_cells(volumes, cells_centre, cells_volume, cell_nodes_range, cell_faces_range)
    cells = Vector{Cell{Float64,SVector{3,Float64},UnitRange{Int64}}}(undef,length(volumes))
    for i = eachindex(volumes)
        cells[i] = Cell(cells_centre[i], cells_volume[i], cell_nodes_range[i], cell_faces_range[i])
    end
    return cells
end

function generate_faces(faces, face_nodes_range, faces_centre, faces_normal, faces_area, face_ownerCells, faces_e, faces_delta, faces_weight)
    faces3D = Vector{Face3D{Float64,SVector{2,Int64},SVector{3,Float64},UnitRange{Int64}}}(undef,length(faces))
    for i = eachindex(faces)
        faces3D[i] = Face3D(face_nodes_range[i], SVector(face_ownerCells[i, 1], face_ownerCells[i, 2]), faces_centre[i], faces_normal[i], faces_e[i], faces_area[i], faces_delta[i], faces_weight[i])
    end
    return faces3D
end