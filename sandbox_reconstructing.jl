using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/800_cell_new_boundaries.unv"

points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

mesh=build_mesh3D(unv_mesh)

mesh.nodes[end].cells_range
mesh.node_cells

mesh.boundaries[end].IDs_range
mesh.boundary_cellsID

mesh.cells[end].nodes_range
mesh.cell_nodes

mesh.cells[end].faces_range
mesh.cell_faces
mesh.cell_neighbours
mesh.cell_nsign

mesh.faces[end].nodes_range
mesh.face_nodes

mesh.nodes
mesh.faces
mesh.cells
mesh.boundaries

