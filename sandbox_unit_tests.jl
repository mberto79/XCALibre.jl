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

mesh,cell_face_nodes=build_mesh3D(unv_mesh)

@time faces_checked, results = check_face_owners(mesh)
@time check_cell_face_nodes(mesh,cell_face_nodes)
boundary_faces(mesh)

function generate_node_cells(points,volumes)
    neighbour=Int64[]
    store=Int64[]
    #cells_range=UnitRange(0,0)
    @inbounds for in=1:length(points)
        @inbounds for iv=1:length(volumes)
            @inbounds for i=1:length(volumes[iv].volumes)
                if volumes[iv].volumes[i]==in
                    neighbour=iv
                    push!(store,neighbour)
                end
                continue
            end
        end
    end
    return store
end

@time node_cells=generate_node_cells(points,volumes)


total=Int[]
mesh.nodes[1].cells_range

node_cells[mesh.nodes[1].cells_range]

node_cells[mesh.nodes[1].cells_range][1]

mesh.cells[node_cells[mesh.nodes[1].cells_range][1]].nodes_range

mesh.cell_nodes[mesh.cells[node_cells[mesh.nodes[1].cells_range][1]].nodes_range]

for in=1:length(mesh.nodes)
    for i=1:length(node_cells[mesh.nodes[in].cells_range])
        if findfirst(x-> x==in,mesh.cell_nodes[mesh.cells[node_cells[mesh.nodes[in].cells_range][i]].nodes_range]) != nothing
            push!(total,1)
            break
        end
    end
end
total
mesh.nodes

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

@time check_node_cells(mesh,node_cells)