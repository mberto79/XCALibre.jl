export compute_geometry!

function compute_geometry!(mesh)
    mesh = calculate_cell_centres!(mesh)
    mesh = calculate_face_centres!(mesh)
    mesh = calculate_face_properties!(mesh)
    mesh = calculate_face_areas!(mesh)
    mesh = calculate_cell_volumes!(mesh)
    return mesh
end

function calculate_cell_centres!(mesh)
    (; nodes, cells, cell_nodes) = mesh
    TF = eltype(nodes[1].coords)
    for (cID, cell) ∈ enumerate(cells)
        sum = SVector{3}(zeros(TF, 3))
        nodesID_range = cell.nodes_range
        nodesID = @view cell_nodes[nodesID_range]
        for nID ∈ nodesID
            sum += nodes[nID].coords
        end
        centre = sum/length(nodesID)
        @reset cell.centre = centre 
        cells[cID] = cell
    end
    return mesh
end

function calculate_face_centres!(mesh)
    (; nodes, faces, face_nodes) = mesh

    TF = eltype(nodes[1].coords)
    for (fID, face) ∈ enumerate(faces)
        sum = SVector{3}(zeros(TF, 3))
        nodesID_range = face.nodes_range
        nodesID = @view face_nodes[nodesID_range]
        for nID ∈ nodesID
            sum += nodes[nID].coords
        end
        centre = sum/length(nodesID)
        @reset face.centre = centre 
        faces[fID] = face
    end
    return mesh
end

function calculate_face_properties!(mesh)
    (; nodes, cells, faces, face_nodes, boundary_cellsID) = mesh
    n_bfaces = length(boundary_cellsID)
    n_faces = length(mesh.faces)

    # loop over boundary faces
    for fID ∈ 1:n_bfaces
        face = faces[fID]
        nIDs = face_nodes[face.nodes_range]
        node1 = nodes[nIDs[1]]
        node2 = nodes[nIDs[2]]
        owners = face.ownerCells

        cell1 = cells[owners[1]]
        fc_n1 = node1.coords - face.centre
        fc_n2 = node2.coords - face.centre 
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

        faces[fID] = face
    end

    # loop over internal faces
    for fID ∈ (n_bfaces + 1):n_faces
        face = faces[fID]
        nIDs = face_nodes[face.nodes_range]
        node1 = nodes[nIDs[1]]
        node2 = nodes[nIDs[2]]
    
        owners = face.ownerCells
        cell1 = cells[owners[1]]
        cell2 = cells[owners[2]]
        fc_n1 = node1.coords - face.centre
        fc_n2 = node2.coords - face.centre 
        cc1_cc2 = cell2.centre - cell1.centre

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
        
        faces[fID] = face
    end
    return mesh
end

function calculate_face_areas!(mesh)
    (; nodes, faces, face_nodes) = mesh 
    for (fID, face) = enumerate(faces)
        nIDs = face_nodes[face.nodes_range]
        extended_nIDs = [nIDs..., nIDs[1]] # TO DO - this is not very efficient
        sum = zero(Float64)
        for ni ∈ eachindex(nIDs)
            nID1 = extended_nIDs[ni]
            nID2 = extended_nIDs[ni+1]
            n1 = nodes[nID1].coords
            n2 = nodes[nID2].coords
            f = face.centre

            n1_n2 = n2 - n1
            n1_f = f - n1
            ec = (n1 + n2)/2 # edge centre
            xf = ec - f
            normal_plane = n1_f × n1_n2
            normal_vec = normal_plane × n1_n2 
            edgeNormal = normal_vec/norm(normal_vec)
            edgeArea = norm(n1_n2)
            sum += xf⋅edgeNormal*edgeArea
        end
        @reset face.area = 0.5*sum
        faces[fID] = face
    end
    return mesh
end

function calculate_cell_volumes!(mesh)
    mesh
end