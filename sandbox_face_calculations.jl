using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK/VTK_writer_3D.jl")


#unv_mesh="src/UNV_3D/TET_PRISM_HM.unv"
# unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/HEXA_HM.unv"

@time mesh = build_mesh3D(unv_mesh)
mesh.faces
mesh.cells
mesh.boundaries

name="tet_prism"

write_vtk(name, mesh::Mesh3)

mesh.cell_faces[mesh.cells[800].faces_range]

points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces
volumes
boundaryElements

calculate_area_and_volume!(mesh) = begin
    (; nodes, faces, face_nodes, cells, cell_faces, cell_nodes) = mesh

    for fID ∈ 1:faces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        if face[fID].nodes_range == 3
            #nIDs = nodeIDs(face_nodes, face.nodes_range)
            n1 = nodes[nIDs[1]]
            n2 = nodes[nIDs[2]]
            n3 = nodes[nIDs[3]]
            
            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2
            
            @reset face.area = area

            faces[fID] = face
        end

        if face[fID].nodes_range >= 4
            n1 = nodes[nIDs[1]]
            n2 = nodes[nIDs[2]]
            n3 = nodes[nIDs[3]]

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2

            for ic=4:length(face[fID].nodes_range)
                n1 = nodes[nIDs[ic]]
                n2 = nodes[nIDs[2]]
                n3 = nodes[nIDs[3]]

                t1x=n2[1]-n1[1]
                t1y=n2[2]-n1[2]
                t1z=n2[3]-n1[3]

                t2x=n3[1]-n1[1]
                t2y=n3[2]-n1[2]
                t2z=n3[3]-n1[3]

                area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
                area=area+sqrt(area2)/2

            end
            @reset face.area = area
            faces[fID] = face
        end
    end


    c = 1/6 # calculate only once
    for cID ∈ eachindex(cells)
        cell = cells[cID]
        # fIDs = faceIDs(cell_faces, cell.faces_range)
        # face = faces[fIDs[1]]
        # nIDs = nodeIDs(face_nodes, face.nodes_range)
        nIDs = nodeIDs(cell_nodes, cell.nodes_range)
        node1 = nodes[nIDs[1]]
        node2 = nodes[nIDs[2]]
        node3 = nodes[nIDs[3]]
        node4 = nodes[nIDs[4]]
        edge1 = node2.coords - node1.coords
        edge2 = node3.coords - node1.coords
        edge3 = node4.coords - node1.coords
        # dist = norm(node2.coords - node1.coords)

        volume = c*((edge1×edge2)⋅edge3)
        @reset cell.volume = volume
        cells[cID] = cell
    end
end
