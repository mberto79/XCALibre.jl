using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"


points, edges, bfaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
bfaces
volumes
boundaryElements

@time mesh = build_mesh3D(unv_mesh)

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
