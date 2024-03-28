using FVM_1D
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


points,edges,bfaces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
bfaces
volumes
boundaryElements

#mesh=build_mesh3D(unv_mesh)
#mesh.nodes

#Priority
#1) all_cell_faces (unsuccsessful)
#2) face_ownerCells (unsuccsessful)
#3) cell neighbours (unsuccsessful)

@time node_cells,cells_range=FVM_1D.UNV_3D.generate_node_cells(points,volumes)

@time nodes=FVM_1D.UNV_3D.generate_nodes(points,cells_range)

@time ifaces,cell_face_nodes=FVM_1D.UNV_3D.generate_tet_internal_faces(volumes,bfaces) #0.065681 seconds
#faces=quad_internal_faces(volumes,faces)

@time face_nodes=FVM_1D.UNV_3D.generate_face_nodes(faces) #0.014925 seconds
@time cell_nodes=FVM_1D.UNV_3D.generate_cell_nodes(volumes) #0.011821 seconds

@time all_cell_faces=FVM_1D.UNV_3D.generate_all_cell_faces(faces,cell_face_nodes) #0.526907 seconds

@time cell_nodes_range=FVM_1D.UNV_3D.generate_cell_nodes_range(volumes) #0.008669 seconds
@time face_nodes_range=FVM_1D.UNV_3D.generate_face_nodes_range(faces) #0.011004 seconds
@time all_cell_faces_range=FVM_1D.UNV_3D.generate_all_cell_faces_range(volumes) #0.010706 seconds

@time cells_centre=FVM_1D.UNV_3D.calculate_centre_cell(volumes,nodes) #0.026527 seconds

@time boundary_faces1,boundary_face_range1=FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements,bfaces) #0.036406 seconds
@time boundary_cells=FVM_1D.UNV_3D.generate_boundary_cells(boundary_faces1,all_cell_faces,all_cell_faces_range) #0.093407 seconds

@time cell_faces,cell_faces_range=FVM_1D.UNV_3D.generate_cell_faces(boundaryElements,volumes,all_cell_faces) #0.055045 seconds

@time boundaries=FVM_1D.UNV_3D.generate_boundaries(boundaryElements,boundary_face_range1) #0.009460 seconds 

@time face_ownerCells= FVM_1D.UNV_3D.generate_face_ownerCells(faces,all_cell_faces,all_cell_faces_range) #0.535271 seconds

@time faces_area=FVM_1D.UNV_3D.calculate_face_area(nodes,faces) #0.037004 seconds
@time faces_centre=FVM_1D.UNV_3D.calculate_face_centre(faces,nodes) #0.026016 seconds
@time faces_normal=FVM_1D.UNV_3D.calculate_face_normal(nodes,faces,face_ownerCells,cells_centre,faces_centre) #0.050983 seconds 
@time faces_e,faces_delta,faces_weight=FVM_1D.UNV_3D.calculate_face_properties(faces,face_ownerCells,cells_centre,faces_centre,faces_normal) #0.061823 seconds

@time cells_volume=FVM_1D.UNV_3D.calculate_cell_volume(volumes,all_cell_faces_range,all_cell_faces,faces_normal,cells_centre,faces_centre,face_ownerCells,faces_area) #0.030618 seconds

@time cells=FVM_1D.UNV_3D.generate_cells(volumes,cells_centre,cells_volume,cell_nodes_range,cell_faces_range) #0.011763 seconds
@time cell_neighbours=FVM_1D.UNV_3D.generate_cell_neighbours(cells,cell_faces) #0.497284 seconds
@time faces=FVM_1D.UNV_3D.generate_faces(faces,face_nodes_range,faces_centre,faces_normal,faces_area,face_ownerCells,faces_e,faces_delta,faces_weight) #0.034309 seconds

@time cell_nsign=FVM_1D.UNV_3D.calculate_cell_nsign(cells,faces,cell_faces) #0.027957 seconds 


#work
bfaces
boundary_faces1
all_cell_faces
all_cell_faces_range
#function generate_boundary_cells(boundary_faces, all_cell_faces, all_cell_faces_range)
    boundary_cells = Int64[]
    store = Int64[]
    for ic = 1:length(bfaces)
        for i in eachindex(all_cell_faces)
            for ip in eachindex(bfaces[ic].faces)
                if all_cell_faces[i] == bfaces[ic].faces[ip]
                    push!(store, i)
                end
            end
        end
    end
    store

    for ic = 1:length(store)
        for i = 1:length(all_cell_faces_range)
            if cell_faces_range[i][1] <= store[ic] <= cell_faces_range[i][end]
                push!(boundary_cells, i)
            end
        end
    end
    return boundary_cells
#end


boundary_faces1
#function generate_boundary_cells(boundary_faces, cell_faces, cell_faces_range)
    boundary_cells = Int64[]
    store = Int64[]
    for ic = 1:length(boundary_faces1)
        for i in eachindex(all_cell_faces)
            if all_cell_faces[i] == boundary_faces1[ic]
                push!(store, boundary_faces1[ic])
            end
        end
    end
    store

    for ic = 1:length(store)
        for i = 1:length(cell_faces_range)
            if cell_faces_range[i][1] <= store[ic] <= cell_faces_range[i][end]
                push!(boundary_cells, i)
            end
        end
    end
    return boundary_cells
#end

bfaces[12]
volumes[1].volumes