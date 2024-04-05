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

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)

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