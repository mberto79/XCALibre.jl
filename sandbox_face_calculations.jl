using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/TET_PRISM_HM.unv"
# unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
# unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"

@time mesh = build_mesh3D(unv_mesh)
# mesh.faces
# mesh.cells
# mesh.boundaries

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

function calculate_face_area(nodes,faces)
    area_store=Float64[]
    for i=1:length(faces)
        if faces[i].faceCount==3
            n1=nodes[faces[i].faces[1]].coords
            n2=nodes[faces[i].faces[2]].coords
            n3=nodes[faces[i].faces[3]].coords

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2
            push!(area_store,area)
        end

        if faces[i].faceCount>3
            n1=nodes[faces[i].faces[1]].coords
            n2=nodes[faces[i].faces[2]].coords
            n3=nodes[faces[i].faces[3]].coords

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2

            for ic=4:faces[i].faceCount
                n1=nodes[faces[i].faces[ic]].coords
                n2=nodes[faces[i].faces[2]].coords
                n3=nodes[faces[i].faces[3]].coords

                t1x=n2[1]-n1[1]
                t1y=n2[2]-n1[2]
                t1z=n2[3]-n1[3]

                t2x=n3[1]-n1[1]
                t2y=n3[2]-n1[2]
                t2z=n3[3]-n1[3]

                area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
                area=area+sqrt(area2)/2

            end

            push!(area_store,area)

        end
    end
    return area_store
end


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