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

mesh,cell_face_nodes,node_cells,all_cell_faces=build_mesh3D(unv_mesh)

@time faces_checked, results = check_face_owners(mesh)
@time check_cell_face_nodes(mesh,cell_face_nodes)
@time boundary_faces(mesh)
@time check_node_cells(mesh,node_cells)
@time check_all_cell_faces(mesh,all_cell_faces)

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

all_cell_faces=generate_all_cell_faces(faces,cell_face_nodes)

function check_all_cell_faces(mesh,all_cell_faces)
    #Check tet cells, no. of faces=4
    #only works for meshes of same cell type
    numface=0
    (; cells,faces)=mesh
    if length(faces[1].nodes_range)==3
        numface=4
    end
    total_cell_faces=length(cells)*numface
    if length(cell_face_nodes)==total_cell_faces
        println("Passed: Length of all_cell_faces matches calculation")
    else
        println("Failed: Warning, length of all_cell_faces does not match calculations")
    end
end

check_all_cell_faces(mesh,all_cell_faces)