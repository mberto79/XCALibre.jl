using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK/VTK_writer_3D.jl")

unv_mesh="src/UNV_3D/TET_PRISM_HM.unv"
unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/HEXA_HM.unv"
unv_mesh="src/UNV_3D/TET_HM.unv"

@time mesh = build_mesh3D(unv_mesh)
# mesh.faces
# mesh.cells
# mesh.boundaries

name="tet_prism"

write_vtk(name, mesh::Mesh3)

d

# points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

# points
# edges
# efaces
# volumes
# boundaryElements

# cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
# node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)  # Should be Hybrid compatible, tested for hexa.
# nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range) # Hyrbid compatible, works for Tet and Hexa
# boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements) # Hybrid compatible

# nbfaces = sum(length.(getproperty.(boundaries, :IDs_range))) # total boundary faces

# bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes) # Hybrid compatible, tested with hexa

# iface_nodes, iface_nodes_range, iface_owners_cells = FVM_1D.UNV_3D.generate_internal_faces(volumes, nbfaces, nodes, node_cells) # Hybrid compatible, tested with hexa.

# bface_nodes,iface_nodes=FVM_1D.UNV_3D.order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes,nodes)

# iface_nodes_range .= [
#     iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
#     ]

# # Concatenate boundary and internal faces
# face_nodes = vcat(bface_nodes, iface_nodes)
# face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
# face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)

# # Sort out cell to face connectivity
# cell_faces, cell_nsign, cell_faces_range, cell_neighbours = FVM_1D.UNV_3D.generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells) # Hybrid compatible. Hexa and tet tested.

# # Build mesh (without calculation of geometry/properties)
# cells = FVM_1D.UNV_3D.build_cells(cell_nodes_range, cell_faces_range) # Hybrid compatible. Hexa tested.
# faces = FVM_1D.UNV_3D.build_faces(face_nodes_range, face_owner_cells) # Hybrid compatible. Hexa tested.

# mesh = Mesh3(
# cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, 
# faces, face_nodes, boundaries, 
# nodes, node_cells,
# SVector{3, Float64}(0.0, 0.0, 0.0), UnitRange{Int64}(0, 0), boundary_cellsID
# ) # Hexa tested.

# # Update mesh to include all geometry calculations required
# FVM_1D.UNV_3D.calculate_centres!(mesh) # Uses centroid instead of geometric. Will need changing, should work fine for regular cells and faces
# FVM_1D.UNV_3D.calculate_face_properties!(mesh) # Touched up face properties, double check values.
# FVM_1D.UNV_3D.calculate_area_and_volume!(mesh)

all_cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(mesh.cells)]
for fID ∈ eachindex(mesh.faces)
    owners = mesh.faces[fID].ownerCells
    owner1 = owners[1]
    owner2 = owners[2]
    #if faces_cpu[fID].ownerCells[1]==cID || faces_cpu[fID].ownerCells[2]==cID
        push!(all_cell_faces[owner1],fID)
        if owner1 !== owner2 #avoid duplication of cells for boundary faces
            push!(all_cell_faces[owner2],fID)
        end
    #end
end
all_cell_faces

for (cID, fIDs) ∈ enumerate(all_cell_faces)
    #write(io,"\t$(length(all_cell_faces[cID]))\n")
    for fID ∈ fIDs

        # write(
        #     io,"\t$(length(faces_cpu[fID].nodes_range)) $(join(face_nodes_cpu[faces_cpu[fID].nodes_range] .- 1," "))\n"
        #     )
    end
end

get_data(array, range, index) = @view array[range[index]]
get_data(array, range) =  array[range] #@view array[range] # 
nodeIDs = get_data
faceIDs = get_data
cellIDs = get_data

segment(p1, p2) = p2 - p1
unit_vector(vec) = vec/norm(vec)

mesh.face_nodes[mesh.faces[2].nodes_range]
mesh.face_nodes
mesh.cell_nodes

nIDs=mesh.face_nodes[mesh.faces[2].nodes_range] # Get ids of nodes of face
    
n1=mesh.nodes[nIDs[1]].coords # Get coords of 4 nodes
n2=mesh.nodes[nIDs[2]].coords
n3=mesh.nodes[nIDs[3]].coords

points = [n1, n2, n3]

_x(n) = n[1]
_y(n) = n[2]
_z(n) = n[3]

l = segment.(Ref(points[1]), points) # surface vectors (segments connecting nodes to reference node)
fn = unit_vector(l[2] × l[3])
fn1 =  unit_vector(cross(l[2], l[3]))
cc=mesh.cells[1].centre
fc=mesh.faces[2].centre
d_fc=fc-cc
dot(d_fc,fn)

if dot(d_fc,fn)<0.0
    nIDs=reverse(nIDs)
end

nIDs

