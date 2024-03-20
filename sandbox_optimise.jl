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


points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

@time generate_tet_internal_faces(volumes,faces)

mesh=build_mesh3D(unv_mesh)

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
    return faces, cell_face_nodes
end

faces,cell_face_nodes=generate_tet_internal_faces(volumes,faces)
faces
cell_face_nodes

# function generate_all_cell_faces(volumes,faces)
#     cell_faces=typeof(faces[1].faceindex)[]
#     for i=1:length(volumes)
#         for ic=1:length(faces)
#             bad=sort(volumes[i].volumes)
#             good=sort(faces[ic].faces)
#             store=typeof(good[1])[]
#             true_store=typeof(true)[]

#             for ip=1:length(good)
#                 push!(store,good[ip] in bad)
#                 push!(true_store,true)
#             end

#             if store[1:length(good)] == true_store
#                 push!(cell_faces,faces[ic].faceindex)
#             end
#             continue
#         end
#     end
#     return cell_faces
# end

@time generate_all_cell_faces(volumes,faces)

cell_face_nodes
sorted_faces

x=setdiff(cell_face_nodes[1],sorted_faces[5])


# faces
# sorted_faces=Vector{Int}[]
# for i=1:length(faces)
#     push!(sorted_faces,sort(faces[i].faces))
# end
# sorted_faces
# cell_face_nodes
# sorted_faces
# all_cell_faces=Int[]
# for i=1:length(cell_face_nodes)
#     for ic=1:length(sorted_faces)
#         x=setdiff(cell_face_nodes[i],sorted_faces[ic])
#         if x==Int64[]
#             push!(all_cell_faces,faces[ic].faceindex)
#         end
#     end
# end
# all_cell_faces




# for i=1:length(cell_face_nodes)
#     x=setdiff(cell_face_nodes,sorted_faces)
#     if x==Int64[]
#         push!(all_cell_faces,faces[i].faceindex)
#     end
# end
# all_cell_faces

all_cell_faces=Int[]
cell_face_nodes

sorted_faces=Vector{Int}[]
for i=1:length(faces)
    push!(sorted_faces,sort(faces[i].faces))
end
sorted_faces

x=findfirst(x -> x==cell_face_nodes[1],sorted_faces)

for i=1:length(cell_face_nodes)
    push!(all_cell_faces,findfirst(x -> x==cell_face_nodes[i],sorted_faces))
end
all_cell_faces


function generate_all_cell_faces(faces,cell_face_nodes)
    all_cell_faces=Int[]
    sorted_faces=Vector{Int}[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end

    for i=1:length(cell_face_nodes)
        push!(all_cell_faces,findfirst(x -> x==cell_face_nodes[i],sorted_faces))
    end
    return all_cell_faces
end

@time cell_faces=generate_all_cell_faces(faces,cell_face_nodes)
