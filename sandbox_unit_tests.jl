using FVM_1D
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

# mesh,cell_face_nodes,node_cells,all_cell_faces=build_mesh3D(unv_mesh)
# mesh.boundaries
# mesh.faces

@time faces_checked, results = check_face_owners(mesh)
@time check_cell_face_nodes(mesh,cell_face_nodes)
@time boundary_faces(mesh)
@time check_node_cells(mesh,node_cells)
@time check_all_cell_faces(mesh,all_cell_faces)

function generate_boundary_faces(boundaryElements)
    boundary_faces=Int64[]
    z=0
    wipe=Int64[]
    boundary_face_range=UnitRange{Int64}[]
    for i=1:length(boundaryElements)
        for n=1:length(boundaryElements[i].elements)
            push!(boundary_faces,boundaryElements[i].elements[n])
            push!(wipe,boundaryElements[i].elements[n])
        end
        if length(wipe)==1
            push!(boundary_face_range,UnitRange(boundaryElements[i].elements[1],boundaryElements[i].elements[1]))
            z=z+1
        elseif length(wipe) ≠1
            push!(boundary_face_range,UnitRange(boundaryElements[i].elements[1],boundaryElements[i].elements[end]))
            z=z+length(wipe)
        end
        wipe=Int64[]
    end
    return boundary_faces,boundary_face_range
end

boundary_faces,boundary_face_range=generate_boundary_faces(boundaryElements)

faces,cell_face_nodes=generate_tet_internal_faces(volumes,faces)
all_cell_faces=generate_all_cell_faces(faces,cell_face_nodes)
all_cell_faces_range=generate_all_faces_range(volumes)
boundary_cells=generate_boundary_cells(boundary_faces,all_cell_faces,all_cell_faces_range)

faces[12]
volumes[1]

function generate_boundary_cells(boundary_faces,cell_faces,cell_faces_range)
    boundary_cells = Int64[]
    store = Int64[]
    for ic=1:length(boundary_faces)
        for i in eachindex(cell_faces)
                if cell_faces[i]==boundary_faces[ic]
                    push!(store,i)
                end
        end
    end
    store
    
    for ic=1:length(store)
        for i=1:length(cell_faces_range)
            if cell_faces_range[i][1]<=store[ic]<=cell_faces_range[i][end]
                push!(boundary_cells,i)
            end
        end
    end
    return boundary_cells
end

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

function generate_all_faces_range(volumes)
    cell_faces_range=UnitRange(0,0)
    store=typeof(cell_faces_range)[]
    x=0
    @inbounds for i=1:length(volumes)
        #Tetra
        if length(volumes[i].volumes)==4
            cell_faces_range=UnitRange(x+1,x+4)
            x=x+4
            push!(store,cell_faces_range)
        end

        #Hexa
        if length(volumes[i].volumes)==8
                cell_faces_range=UnitRange(x+1,x+6)
                x=x+6
                push!(store,cell_faces_range)
        end
    end
    return store
end