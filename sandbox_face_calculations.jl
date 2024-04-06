using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/TET_PRISM.unv"
unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"

@time mesh = build_mesh3D(unv_mesh)
mesh.faces
# mesh.cells
# mesh.boundaries

points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces
volumes
boundaryElements

calculate_face_properties!(mesh) = begin
    # written for Tet only for debugging
    (; nodes, cells, faces, face_nodes, boundary_cellsID) = mesh
    n_bfaces = length(boundary_cellsID)
    n_faces = length(mesh.faces)

    # loop over boundary faces
    for fID ∈ 1:n_bfaces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        node1 = nodes[nIDs[1]]
        node2 = nodes[nIDs[2]]
        node3 = nodes[nIDs[3]]
        edge1 = node2.coords - node1.coords
        edge2 = node3.coords - node1.coords
        owners = face.ownerCells

        cell1 = cells[owners[1]]
        # cell2 = cells[owners[2]]
        fc_n1 = node1.coords - face.centre
        fc_n2 = node2.coords - face.centre 
        # cc1_cc2 = cell2.centre - cell1.centre
        cc1_cc2 = face.centre - cell1.centre
        normal_vec = fc_n1 × fc_n2
        normal = normal_vec/norm(normal_vec)
        if cc1_cc2 ⋅ normal < 0
            normal *= -1
        end
        @reset face.normal = normal

        # delta
        cc_fc = face.centre - cell1.centre
        delta = norm(cc_fc)
        e = cc_fc/delta
        weight = one(Float64)
        @reset face.delta = delta
        @reset face.e = e
        @reset face.weight = weight
        #@reset face.area = 0.5*norm(edge1×edge2)

        faces[fID] = face
    end

    # loop over internal faces
    for fID ∈ (n_bfaces + 1):n_faces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        node1 = nodes[nIDs[1]]
        node2 = nodes[nIDs[2]]
        node3 = nodes[nIDs[3]]
        edge1 = node2.coords - node1.coords
        edge2 = node3.coords - node1.coords
        owners = face.ownerCells
        cell1 = cells[owners[1]]
        cell2 = cells[owners[2]]
        fc_n1 = node1.coords - face.centre
        fc_n2 = node2.coords - face.centre 
        cc1_cc2 = cell2.centre - cell1.centre
        # cc1_cc2 = face.centre - cell1.centre
        normal_vec = fc_n1 × fc_n2
        normal = normal_vec/norm(normal_vec)
        if cc1_cc2 ⋅ normal < 0
            normal *= -1
        end
        @reset face.normal = normal

        # delta
        c1_c2 = cell2.centre - cell1.centre
        fc_c1 = face.centre - cell1.centre
        c2_fc = cell2.centre - face.centre
        delta = norm(c1_c2)
        e = c1_c2/delta
        weight = abs((fc_c1 ⋅ normal)/((fc_c1 ⋅ normal)+(c2_fc ⋅ normal)))
        @reset face.delta = delta
        @reset face.e = e
        @reset face.weight = weight
        #@reset face.area = 0.5*norm(edge1×edge2)
        
        faces[fID] = face
    end
end

#Old Function
function calculate_face_properties(faces,face_ownerCells,cell_centre,face_centre,face_normal)
    store_e=[]
    store_delta=[]
    store_weight=[]
    for i=1:length(faces)
        if face_ownerCells[i,2]==face_ownerCells[i,1]
            cc=cell_centre[face_ownerCells[i,1]]
            cf=face_centre[i]

            d_cf=cf-cc

            delta=norm(d_cf)
            push!(store_delta,delta)
            e=d_cf/delta
            push!(store_e,e)
            weight=one(Float64)
            push!(store_weight,weight)

        else
            c1=cell_centre[face_ownerCells[i,1]]
            c2=cell_centre[face_ownerCells[i,2]]
            cf=face_centre[i]
            d_1f=cf-c1
            d_f2=c2-cf
            d_12=c2-c1

            delta=norm(d_12)
            push!(store_delta,delta)
            e=d_12/delta
            push!(store_e,e)
            weight=abs((d_1f⋅face_normal[i])/(d_1f⋅face_normal[i] + d_f2⋅face_normal[i]))
            push!(store_weight,weight)

        end
    end
    return store_e,store_delta,store_weight
end