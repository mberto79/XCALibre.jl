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

function generate_cell_nodes(volumes)
    #cell_nodes = Vector{Int64}(undef, length(volumes) * 4) #length of cells times number of nodes per cell (tet only)
    cell_nodes = Int64[] # cell_node length is undetermined as mesh could be hybrid, using push. Could use for and if before to preallocate vector.
    counter = 0
    for n = eachindex(volumes)
        for i = 1:volumes[n].volumeCount
            counter = counter + 1
            push!(cell_nodes,volumes[n].volumes[i])
        end
    end
    cell_nodes

    cell_nodes_range = Vector{UnitRange{Int64}}(undef, length(volumes)) # cell_nodes_range determined by no. of cells.
    x = 0
    for i = eachindex(volumes)
        cell_nodes_range[i] = UnitRange(x + 1, x + length(volumes[i].volumes))
        x = x + length(volumes[i].volumes)
    end
    return cell_nodes, cell_nodes_range
end