using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")

unv_mesh = "src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh = "unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.08m.unv"
unv_mesh = "unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.06m.unv"
unv_mesh = "unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.04m.unv"
unv_mesh = "unv_sample_meshes/3d_streamtube_0.5x0.1x0.1_0.03m.unv"

unv_mesh = "unv_sample_meshes/3D_cylinder.unv"

#unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"


points, edges, bfaces, volumes, boundaryElements = load_3D(unv_mesh, scale=1, integer=Int64, float=Float64)

points
edges
bfaces
volumes
boundaryElements

@time mesh = build_mesh3D(unv_mesh)

_boundary_faces(mesh)
check_face_owners(mesh)
check_cell_face_nodes(mesh)
check_node_cells(mesh)
check_all_cell_faces(mesh)
check_boundary_faces(mesh)
#mesh.nodes

#Priority
#1) all_cell_faces (unsuccsessful)
#2) face_ownerCells (unsuccsessful)
#3) cell neighbours (unsuccsessful)

@time cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) 
@time node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes) 
@time nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range)
@time boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements) 

@time bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = 
begin
    FVM_1D.UNV_3D.generate_boundary_faces(
        boundaryElements, bfaces, node_cells,node_cells_range, volumes)
end

@time iface_nodes, iface_nodes_range, iface_owners_cells = 
begin
    FVM_1D.UNV_3D.generate_internal_faces(volumes, bfaces, nodes, node_cells)
end

# shift range
@time iface_nodes_range .= [
    iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
    ]

@time face_nodes = vcat(bface_nodes, iface_nodes)
@time face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
@time face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)

@time cell_faces, cell_nsign, cell_faces_range, cell_neighbours = begin
    FVM_1D.UNV_3D.generate_cell_face_connectivity(volumes, bfaces, face_owner_cells)
end

@time cells = FVM_1D.UNV_3D.build_cells(cell_nodes_range, cell_faces_range)
@time faces = FVM_1D.UNV_3D.build_faces(face_nodes_range, face_owner_cells)

@time mesh = Mesh3(
            cells, 
            cell_nodes, 
            cell_faces, 
            cell_neighbours, 
            cell_nsign, 
            faces, 
            face_nodes, 
            boundaries, 
            nodes, 
            node_cells,
            SVector{3, Float64}(0.0, 0.0, 0.0), # get_float
            UnitRange{Int64}(0, 0), # get_int
            boundary_cellsID
        )

@time FVM_1D.UNV_3D.calculate_centres!(mesh)
@time FVM_1D.UNV_3D.calculate_face_properties!(mesh)
@time FVM_1D.UNV_3D.calculate_area_and_volume!(mesh)





@time cells_centre = FVM_1D.UNV_3D.calculate_centre_cell(volumes, nodes) #0.026527 seconds
@time faces_area = FVM_1D.UNV_3D.calculate_face_area(nodes, faces) #0.037004 seconds
@time faces_centre = FVM_1D.UNV_3D.calculate_face_centre(faces, nodes) #0.026016 seconds
@time faces_normal = FVM_1D.UNV_3D.calculate_face_normal(nodes, faces, face_ownerCells, cells_centre, faces_centre) #0.050983 seconds 
@time faces_e, faces_delta, faces_weight = FVM_1D.UNV_3D.calculate_face_properties(faces, face_ownerCells, cells_centre, faces_centre, faces_normal) #0.061823 seconds

@time cells_volume = FVM_1D.UNV_3D.calculate_cell_volume(volumes, all_cell_faces_range, all_cell_faces, faces_normal, cells_centre, faces_centre, face_ownerCells, faces_area) #0.030618 seconds


@time cells = FVM_1D.UNV_3D.generate_cells(volumes, cells_centre, cells_volume, cell_nodes_range, cell_faces_range) #0.011763 seconds
@time cell_neighbours = FVM_1D.UNV_3D.generate_cell_neighbours(cells, cell_faces) #0.497284 seconds
@time faces = FVM_1D.UNV_3D.generate_faces(faces, face_nodes_range, faces_centre, faces_normal, faces_area, face_ownerCells, faces_e, faces_delta, faces_weight) #0.034309 seconds

#work
function calculate_cell_nsign(cells, faces, cell_faces)
    cell_nsign = Vector{Int}(undef, length(cell_faces))
    counter = 0
    for i = 1:length(cells)
        centre = cells[i].centre
        for ic = 1:length(cells[i].faces_range)
            fcentre = faces[cell_faces[cells[i].faces_range][ic]].centre
            fnormal = faces[cell_faces[cells[i].faces_range][ic]].normal
            d_cf = fcentre - centre
            fnsign = zero(Int)

            if d_cf ⋅ fnormal > zero(Float64)
                fnsign = one(Int)
            else
                fnsign = -one(Int)
            end
            counter = counter + 1
            cell_nsign[counter] = fnsign
        end

    end
    return cell_nsign
end

calculate_cell_nsign(cells, faces, cell_faces)