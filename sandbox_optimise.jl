using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.08m.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.06m.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.04m.unv"
#unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"


# points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

# points
# edges
# faces
# volumes
# boundaryElements

@time generate_tet_internal_faces(volumes,faces)


mesh=build_mesh3D(unv_mesh)

#function generate_internal_faces(volumes,faces)
    store_cell_faces=Int64[]
    store_faces=Int64[]
    
    for i=1:length(volumes)
        cell=sort(volumes[i].volumes)
        push!(store_cell_faces,cell[1],cell[2],cell[3])
        push!(store_cell_faces,cell[1],cell[2],cell[4])
        push!(store_cell_faces,cell[1],cell[3],cell[4])
        push!(store_cell_faces,cell[2],cell[3],cell[4])
    end
    store_cell_faces
    
    for i=1:length(faces)
        face=sort(faces[i].faces)
        push!(store_faces,face[1],face[2],face[3])
    end
    store_faces

    range=UnitRange{Int64}[]

    x=0
    for i=1:length(store_cell_faces)/3 # Very dodgy! could give float!!!
        store=UnitRange(x+1:x+3)
        x=x+3
        push!(range,store)
    end
    range

    faces1=Vector{Int64}[]

    for i=1:length(range)
        face1=store_cell_faces[range[i]]
        for ic=1:length(range)
            face2=store_cell_faces[range[ic]]
            store=[] # Why are you allocating inside the loop?
    
            push!(store,face2[1] in face1)
            push!(store,face2[2] in face1)
            push!(store,face2[3] in face1)

            count1=count(store)
            # println(count1)
    
            if count1!=3
            # if length(store) !=3 # is this what you want to do?
                push!(faces1,face1)
            end
        end
    end

    all_faces=unique(faces1)

    store1_faces=Vector{Int64}[]
    for i=1:length(faces)
        push!(store1_faces,sort(faces[i].faces))
    end

    all_faces=sort(all_faces)
    store1_faces=sort(store1_faces)
    
    internal_faces=setdiff(all_faces,store1_faces)
    
    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
#end

@time generate_internal_faces(volumes,faces)



function generate_cell_face_nodes(volumes)
    cell_face_nodes=Int64[]
    for i=1:length(volumes)
        cell=sort(volumes[i].volumes)
        push!(cell_face_nodes,cell[1],cell[2],cell[3])
        push!(cell_face_nodes,cell[1],cell[2],cell[4])
        push!(cell_face_nodes,cell[1],cell[3],cell[4])
        push!(cell_face_nodes,cell[2],cell[3],cell[4])
    end
    return cell_face_nodes
end

cell_face_nodes=generate_cell_face_nodes(volumes)

function generate_bface_nodes(faces)
    bface_nodes=Int64[]
    for i=1:length(faces)
        face=sort(faces[i].faces)
        push!(bface_nodes,face[1],face[2],face[3])
    end
    return bface_nodes
end

bface_nodes=generate_bface_nodes(faces)

# range=UnitRange{Int64}[]

# x=0
# for i=1:length(volumes)*length(volumes[1].volumes) #same length as no. of faces in cell x no. of cells
#     store=UnitRange(x+1:x+3)
#     x=x+3
#     push!(range,store)
# end
# range

cell_face_nodes=Vector{Int}[]

for i=1:length(volumes)
    cell_faces=zeros(Int,4,3)
    cell=sort(volumes[i].volumes)

    cell_faces[1,1:3]=cell[1:3]
    cell_faces[2,1:2]=cell[1:2]
    cell_faces[2,3]=cell[4]
    cell_faces[3,1]=cell[1]
    cell_faces[3,2:3]=cell[3:4]
    cell_faces[4,1:3]=cell[2:4]

    for ic=1:4
        push!(cell_face_nodes,cell_faces[ic,:])
    end
end
cell_face_nodes

sorted_faces=Vector{Int}[]
for i=1:length(faces)
    push!(sorted_faces,sort(faces[i].faces))
end
sorted_faces

internal_faces=setdiff(cell_face_nodes,sorted_faces)

for i=1:length(internal_faces)
    push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
end
return faces

function generate_tet_internal_faces(volumes,faces)
    cell_face_nodes=Vector{Int}[]

    for i=1:length(volumes)
        cell_faces=zeros(Int,4,3)
        cell=sort(volumes[i].volumes)

        cell_faces[1,1:3]=cell[1:3]
        cell_faces[2,1:2]=cell[1:2]
        cell_faces[2,3]=cell[4]
        cell_faces[3,1]=cell[1]
        cell_faces[3,2:3]=cell[3:4]
        cell_faces[4,1:3]=cell[2:4]

        for ic=1:4
            push!(cell_face_nodes,cell_faces[ic,:])
        end
    end

    sorted_faces=Vector{Int}[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end

    internal_faces=setdiff(cell_face_nodes,sorted_faces)

    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
end

@time generate_tet_internal_faces(volumes,faces)



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

@time quad_internal_faces(volumes,faces)