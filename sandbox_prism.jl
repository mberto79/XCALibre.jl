using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/TET_PRISM_HM.unv"
unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/HEXA_HM.unv"
unv_mesh="src/UNV_3D/TET_HM.unv"

@time mesh = build_mesh3D(unv_mesh)
# mesh.faces
# mesh.cells
# mesh.boundaries

name="tet_prism"

write_vtk(name, mesh::Mesh3)

points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces
volumes
boundaryElements

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes)
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)
nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range)
boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements)

nbfaces = sum(length.(getproperty.(boundaries, :IDs_range)))

bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes)

bface_nodes_range
bface_nodes

iface_nodes, iface_nodes_range, iface_owners_cells = FVM_1D.UNV_3D.generate_internal_faces(volumes, nbfaces, nodes, node_cells)

iface_nodes_range
iface_nodes

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


cell_faces, cell_nsign, cell_faces_range, cell_neighbours=FVM_1D.UNV_3D.generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells)

cell_faces
cell_faces_range
cell_nsign
cell_neighbours

length(cell_faces_range[723])

cells = FVM_1D.UNV_3D.build_cells(cell_nodes_range, cell_faces_range)

faces = FVM_1D.UNV_3D.build_faces(face_nodes_range, face_owner_cells)

