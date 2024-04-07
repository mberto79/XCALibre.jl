using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK/VTK_writer_3D.jl")

unv_mesh="src/UNV_3D/TET_PRISM_HM.unv"
unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/HEXA_HM.unv"
unv_mesh="src/UNV_3D/TET_HM.unv"

@time mesh = build_mesh3D(unv_mesh)
mesh.faces
mesh.cells
mesh.boundaries

points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces
volumes
boundaryElements

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)  # Should be Hybrid compatible, tested for hexa.
nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range) # Hyrbid compatible, works for Tet and Hexa
boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements) # Hybrid compatible

nbfaces = sum(length.(getproperty.(boundaries, :IDs_range))) # total boundary faces

bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = 
begin
    FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes) # Hybrid compatible, tested with hexa
end

iface_nodes, iface_nodes_range, iface_owners_cells = 
begin 
    FVM_1D.UNV_3D.generate_internal_faces(volumes, nbfaces, nodes, node_cells) # Hybrid compatible, tested with hexa.
end

bface_nodes,iface_nodes=FVM_1D.UNV_3D.order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes,nodes)

get_data(array, range, index) = @view array[range[index]]
get_data(array, range) =  array[range] #@view array[range] # 
nodeIDs = get_data
faceIDs = get_data
cellIDs = get_data

segment(p1, p2) = p2 - p1
angle1(s, i1, i2) = acosd( (s[i1]⋅s[i2])/(norm(s[i1])*norm(s[i2])))
unit_vector(vec) = vec/norm(vec)

bface_nodes_range
iface_nodes_range

n_bfaces = length(bface_nodes_range)
n_ifaces =  length(iface_nodes_range)

order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes)

function order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes)
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
    
            #fc=sum(points)/length(points) # geographic centre (not centroid)
            #s = segment.(Ref(pc), points) # surface vectors (from face centre)
            #u = unit_vector.(s)
            l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
            fn = unit_vector(l[2] × l[3]) # face normal vector
    
            angles=Float64[] # Vector to store angles
            theta2 = angle1(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
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
    
            #fc=sum(points)/length(points) # geographic centre (not centroid)
            #s = segment.(Ref(pc), points) # surface vectors (from face centre)
            #u = unit_vector.(s)
            l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
            fn = unit_vector(l[2] × l[3]) # face normal vector
    
            angles=Float64[]
            theta2 = angle1(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
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



for fID = 1:n_bfaces # Re-order Boundary Faces
    if bface_nodes_range[fID] == 4 # Only for Quad faces
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
        s = segment.(Ref(pc), points) # surface vectors (from face centre)
        u = unit_vector.(s)
        l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
        fn = unit_vector(l[2] × l[3]) # face normal vector

        angles=Float64[] # Vector to store angles
        theta2 = angle(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
        theta3 = angle(l, 2, 3)*(signbit((l[2] × fn)⋅l[3]) ? 1 : -1)
        theta4 = angle(l, 2, 4)*(signbit((l[2] × fn)⋅l[4]) ? 1 : -1)

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
bface_nodes

for fID = 1:n_ifaces # Re-order internal faces
    if iface_nodes_range[fID]==4 # Only for Quad Faces
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
        s = segment.(Ref(pc), points) # surface vectors (from face centre)
        u = unit_vector.(s)
        l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
        fn = unit_vector(l[2] × l[3]) # face normal vector

        angles=Float64[]
        theta2 = angle(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
        theta3 = angle(l, 2, 3)*(signbit((l[2] × fn)⋅l[3]) ? 1 : -1)
        theta4 = angle(l, 2, 4)*(signbit((l[2] × fn)⋅l[4]) ? 1 : -1)

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
iface_nodes

function order_face_nodes()
    n_bfaces = length(bface_nodes_range)
    n_ifaces =  length(iface_nodes_range)
    for fID = 1:n_bfaces # Re-order Boundary Faces
        if bface_nodes_range[fID] == 4 # Only for Quad faces
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
            s = segment.(Ref(pc), points) # surface vectors (from face centre)
            u = unit_vector.(s)
            l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
            fn = unit_vector(l[2] × l[3]) # face normal vector
    
            angles=Float64[] # Vector to store angles
            theta2 = angle(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
            theta3 = angle(l, 2, 3)*(signbit((l[2] × fn)⋅l[3]) ? 1 : -1)
            theta4 = angle(l, 2, 4)*(signbit((l[2] × fn)⋅l[4]) ? 1 : -1)
    
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
        if iface_nodes_range[fID]==4 # Only for Quad Faces
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
            s = segment.(Ref(pc), points) # surface vectors (from face centre)
            u = unit_vector.(s)
            l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
            fn = unit_vector(l[2] × l[3]) # face normal vector
    
            angles=Float64[]
            theta2 = angle(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
            theta3 = angle(l, 2, 3)*(signbit((l[2] × fn)⋅l[3]) ? 1 : -1)
            theta4 = angle(l, 2, 4)*(signbit((l[2] × fn)⋅l[4]) ? 1 : -1)
    
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


#

p1 = [0.0,0.0,0.0]
p2 = [1.0,1.0,0.0]
p3 = [1.0,0.0,0.0]
p4 = [0,1.0,0.0]

points = [p1, p2, p3, p4]

_x(p) = p[1]
_y(p) = p[2]
_z(p) = p[3]


pc = sum(points)/length(points) # geographic centre (not centroid)


segment(p1, p2) = p2 - p1
angle(s, i1, i2) = acosd( (s[i1]⋅s[i2])/(norm(s[i1])*norm(s[i2])))
unit_vector(vec) = vec/norm(vec)


# This will need to be rewritten so that "l" does not return a segment with the reference point itself which, of course is 0.
points
s = segment.(Ref(pc), points) # surface vectors (from face centre)
u = unit_vector.(s)
l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)


fn = unit_vector(l[2] × l[3]) # face normal vector


#theta = angle(l, 2, 1)*(signbit((l[2] × fn)⋅l[1]) ? 1 : -1)
theta = angle(l, 2, 2)*(signbit((l[2] × fn)⋅l[2]) ? 1 : -1)
theta = angle(l, 2, 3)*(signbit((l[2] × fn)⋅l[3]) ? 1 : -1)
theta = angle(l, 2, 4)*(signbit((l[2] × fn)⋅l[4]) ? 1 : -1)

