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

get_data(array, range, index) = @view array[range[index]]
get_data(array, range) =  array[range] #@view array[range] # 
nodeIDs = get_data
faceIDs = get_data
cellIDs = get_data

function generate_boundary_faces(
    boundaryElements, bfaces, node_cells, node_cells_range, volumes
    )
    bface_nodes = Vector{Vector{Int64}}(undef, length(bfaces))
    bface_nodes_range = Vector{UnitRange{Int64}}(undef,length(bfaces))
    bowners_cells = Vector{Int64}[Int64[0,0] for _ ∈ eachindex(bfaces)]
    boundary_cells = Vector{Int64}(undef,length(bfaces))

    fID = 0 # faceID index of output array (reordered)
    start = 1
    for boundary ∈ boundaryElements
        elements = boundary.elements
            for bfaceID ∈ elements
                fID += 1
                nnodes = length(bfaces[bfaceID].faces)
                nodeIDs = bfaces[bfaceID].faces # Actually nodesIDs
                bface_nodes[fID] = nodeIDs
                bface_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
                start += nnodes

                # Find owner cells (same as boundary cells)
                assigned = false
                for nodeID ∈ nodeIDs
                    cIDs = cellIDs(node_cells, node_cells_range, nodeID)
                    for cID ∈ cIDs
                        if intersect(nodeIDs, volumes[cID].volumes) == nodeIDs
                            bowners_cells[fID] .= cID
                            boundary_cells[fID] = cID
                            assigned = true
                            break
                        end
                    end
                    if assigned 
                        break
                    end
                end
            end
    end

    bface_nodes = vcat(bface_nodes...) # Flatten - from Vector{Vector{Int}} to Vector{Int}
    # Check: Length of bface_nodes = no. of bfaces x no. of nodes of each bface

    return bface_nodes, bface_nodes_range, bowners_cells, boundary_cells
end

bface_nodes
bface_nodes_range
bowners_cells
boundary_cells