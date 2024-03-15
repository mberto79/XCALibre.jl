using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra
include("src/VTK_3D/VTU.jl")

#unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/800_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"

points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

mesh=build_mesh3D(unv_mesh)
mesh.boundaries
mesh.faces
name="test_vtu"
write_vtu(name,mesh)

faces

mesh.face_nodes[mesh.faces[1].nodes_range]
mesh.cells
mesh.cell_neighbours




#function quad_internal_faces(volumes,faces)
    store_cell_faces=[]
    store_cell_faces1=[]
    for i=1:length(volumes)
            #cell=sort(volumes[i].volumes)
            push!(store_cell_faces,volumes[i].volumes[1],volumes[i].volumes[2],volumes[i].volumes[3])
            #push!(store_cell_faces,cell[1],cell[2],cell[4])
            push!(store_cell_faces,volumes[i].volumes[1],volumes[i].volumes[2],volumes[i].volumes[4])
            #push!(store_cell_faces,cell[1],cell[3],cell[4])
            push!(store_cell_faces,volumes[i].volumes[1],volumes[i].volumes[3],volumes[i].volumes[4])
            #push!(store_cell_faces,cell[2],cell[3],cell[4])
            push!(store_cell_faces,volumes[i].volumes[2],volumes[i].volumes[3],volumes[i].volumes[4])
    end


    store_cell_faces

    store_cell_faces=[]
    store_cell_faces1=[]
    store_cell_ID=[]

    for i=1:length(volumes)
        cell_faces=zeros(Int,4,3)

        cell_faces[1,1:3]=volumes[i].volumes[1:3]
        cell_faces[2,1:2]=volumes[i].volumes[1:2]
        cell_faces[2,3]=volumes[i].volumes[4]
        cell_faces[3,1]=volumes[i].volumes[1]
        cell_faces[3,2:3]=volumes[i].volumes[3:4]
        cell_faces[4,1:3]=volumes[i].volumes[2:4]

        for ic=1:4
            push!(store_cell_faces1,cell_faces[ic,:])
            push!(store_cell_ID,i)
        end
    end

    faces
    store_cell_faces1
    store_cell_ID

    sorted_cell_faces=[]
    for i=1:length(store_cell_faces1)

        push!(sorted_cell_faces,sort(store_cell_faces1[i]))
    end

    sorted_cell_faces

    faces
    sorted_faces=[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end
    sorted_faces

    internal_faces=setdiff(sorted_cell_faces,sorted_faces)

    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
#end


# #function generate_internal_faces(volumes,faces)
#     store_cell_faces=Int[]
#     store_faces=Int[]
    
#     for i=1:length(volumes)
#     cell=sort(volumes[i].volumes)
#     push!(store_cell_faces,cell[1],cell[2],cell[3])
#     push!(store_cell_faces,cell[1],cell[2],cell[4])
#     push!(store_cell_faces,cell[1],cell[3],cell[4])
#     push!(store_cell_faces,cell[2],cell[3],cell[4])
#     end
    
#     store_cell_faces
#     for i=1:length(faces)
#         face=sort(faces[i].faces)
#         push!(store_faces,face[1],face[2],face[3])
#     end

#     range=[]

#     x=0
#     for i=1:length(store_cell_faces)/3
#         store=UnitRange(x+1:x+3)
#         x=x+3
#         push!(range,store)
#     end

#     faces1=[]

#     for i=1:length(range)
#         face1=store_cell_faces[range[i]]
#         for ic=1:length(range)
#             face2=store_cell_faces[range[ic]]
#             store=[]
    
#             push!(store,face2[1] in face1)
#             push!(store,face2[2] in face1)
#             push!(store,face2[3] in face1)
    
#             count1=count(store)
    
#             if count1!=3
#                 push!(faces1,face1)
#             end
#         end
#     end
#     faces1

#     all_faces=unique(faces1)

#     store1_faces=[]
#     for i=1:length(faces)
#         push!(store1_faces,sort(faces[i].faces))
#     end

#     all_faces=sort(all_faces)
#     store1_faces=sort(store1_faces)
    
#     internal_faces=setdiff(all_faces,store1_faces)
    
#     for i=1:length(internal_faces)
#         push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
#     end
#     return faces
#end

function generate_nodes(points,volumes)
    nodes=Node{SVector{3,Float64}, UnitRange{Int64}}[]
    cells_range=nodes_cells_range!(points,volumes)
    @inbounds for i ∈ 1:length(points)
        #point=points[i].xyz
        push!(nodes,Node(points[i].xyz,cells_range[i]))
    end
    return nodes
end

nodes=generate_nodes(points,volumes)

function nodes_cells_range!(points,volumes)
    neighbour=Int64[]
    wipe=Int64[]
    cells_range=UnitRange[]
    x=0
    @inbounds for in=1:length(points)
        @inbounds for iv=1:length(volumes)
            @inbounds for i=1:length(volumes[iv].volumes)
                if volumes[iv].volumes[i]==in
                    neighbour=iv
                    push!(wipe,neighbour)
                    
                end
                continue
                
            end
        end
        if length(wipe)==1
            #cells_range[in]=UnitRange(x+1,x+1)
            push!(cells_range,UnitRange(x+1,x+1))
            x=x+1
        elseif length(wipe) ≠1
            #cells_range[in]=UnitRange(x+1,x+length(wipe))
            push!(cells_range,UnitRange(x+1,x+length(wipe)))
            x=x+length(wipe)
        end
        #push!(mesh.nodes[in].cells_range,cells_range)
        wipe=Int64[]
    end
    return cells_range
end

#function quad_internal_faces(volumes,faces)
    store_cell_faces1=[]

    for i=1:length(volumes)
        cell_faces=zeros(Int,6,4)

        cell_faces[1,1:4]=volumes[i].volumes[1:4]
        cell_faces[2,1:4]=volumes[i].volumes[5:8]
        cell_faces[3,1:2]=volumes[i].volumes[1:2]
        cell_faces[3,3:4]=volumes[i].volumes[5:6]
        cell_faces[4,1:2]=volumes[i].volumes[3:4]
        cell_faces[4,3:4]=volumes[i].volumes[7:8]
        cell_faces[5,1:2]=volumes[i].volumes[2:3]
        cell_faces[5,3:4]=volumes[i].volumes[6:7]
        cell_faces[6,1]=volumes[i].volumes[1]
        cell_faces[6,2]=volumes[i].volumes[4]
        cell_faces[6,3]=volumes[i].volumes[5]
        cell_faces[6,4]=volumes[i].volumes[8]

        for ic=1:6
            push!(store_cell_faces1,cell_faces[ic,:])
        end
    end

    store_cell_faces1

    sorted_cell_faces=[]
    for i=1:length(store_cell_faces1)

        push!(sorted_cell_faces,sort(store_cell_faces1[i]))
    end

    sorted_cell_faces

    faces
    sorted_faces=[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end
    sorted_faces

    internal_faces=setdiff(sorted_cell_faces,sorted_faces)

    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
#end

#quad_internal_faces(volumes,faces)

p1=nodes[faces[1].faces[1]].coords
p2=nodes[faces[1].faces[2]].coords
p3=nodes[faces[1].faces[3]].coords
p4=nodes[faces[1].faces[4]].coords

fc=(p1+p2+p3+p4)/4

fn=((p1-fc) × (p2-fc))/norm((p1-fc) × (p2-fc))


((p1-fc)×(p2-fc)) ⋅ fn
((p1-fc)×(p3-fc)) ⋅ fn
((p1-fc)×(p4-fc)) ⋅ fn



function calculate_face_normal(faces,nodes)
    normal_store=[]
    for i=1:length(faces)
        n1=nodes[faces[i].faces[1]].coords
        n2=nodes[faces[i].faces[2]].coords
        n3=nodes[faces[i].faces[3]].coords

        t1x=n2[1]-n1[1]
        t1y=n2[2]-n1[2]
        t1z=n2[3]-n1[3]

        t2x=n3[1]-n1[1]
        t2y=n3[2]-n1[2]
        t2z=n3[3]-n1[3]

        nx=t1y*t2z-t1z*t2y
        ny=-(t1x*t2z-t1z*t2x)
        nz=t1x*t2y-t1y*t2x

        magn2=(nx)^2+(ny)^2+(nz)^2

        snx=nx/sqrt(magn2)
        sny=ny/sqrt(magn2)
        snz=nz/sqrt(magn2)

        normal=SVector(snx,sny,snz)
        push!(normal_store,normal)
    end
    return normal_store
end

calculate_face_normal(faces,nodes)