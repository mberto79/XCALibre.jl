using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"


points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces
volumes
boundaryElements

#@time mesh = build_mesh3D(unv_mesh)

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)
nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range)
boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements)

nbfaces = sum(length.(getproperty.(boundaries, :IDs_range)))

bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes)

iface_nodes, iface_nodes_range, iface_owners_cells = FVM_1D.UNV_3D.generate_internal_faces(volumes, nbfaces, nodes, node_cells)


get_data(array, range, index) = @view array[range[index]]
get_data(array, range) =  array[range] #@view array[range] # 
nodeIDs = get_data
faceIDs = get_data
cellIDs = get_data

iface_nodes_range .= [
iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
]

# Concatenate boundary and internal faces
face_nodes = vcat(bface_nodes, iface_nodes)
face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)


#function generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells)
    cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(volumes)] 
    cell_nsign = Vector{Int64}[Int64[] for _ ∈ eachindex(volumes)] 
    cell_neighbours = Vector{Int64}[Int64[] for _ ∈ eachindex(volumes)] 
    cell_faces_range = UnitRange{Int64}[UnitRange{Int64}(0,0) for _ ∈ eachindex(volumes)] 

    # Pass face ID to each cell
    first_internal_face = nbfaces + 1
    total_faces = length(face_owner_cells)
    for fID ∈ first_internal_face:total_faces
        owners = face_owner_cells[fID] # 2 cell owners IDs
        owner1 = owners[1]
        owner2 = owners[2]
        push!(cell_faces[owner1], fID) # Only for internal faces.
        push!(cell_faces[owner2], fID)
        push!(cell_nsign[owner1], 1) # Contract: Face normal goes from owner 1 to 2      
        push!(cell_nsign[owner2], -1)   
        push!(cell_neighbours[owner1], owner2)     
        push!(cell_neighbours[owner2], owner1)     
    end
    cell_faces
    cell_nsign
    cell_neighbours

    # Generate cell faces range
    start = 1
    for (cID, faces) ∈ enumerate(cell_faces)
        nfaces = length(faces)
        cell_faces_range[cID] = UnitRange{Int64}(start:(start + nfaces - 1))
        start += nfaces
    end
cell_faces_range
    # Flatten output (go from Vector{Vector{Int}} to Vector{Int})

    cell_faces = vcat(cell_faces...)
    cell_nsign = vcat(cell_nsign...)

    # NOTE: Check how this is being accessed in RANS models (need to flatten?)
    cell_neighbours = vcat(cell_neighbours...) # Need to check RANSMODELS!!!

    return cell_faces, cell_nsign, cell_faces_range, cell_neighbours
end