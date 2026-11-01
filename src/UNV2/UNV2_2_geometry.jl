function centre2d(face::Face2D{I,F}) where {I,F}
    c = face.centre
    [c[1]], [c[2]]
end

function centre2d(cell::Cell{I,F}) where {I,F}
    c = cell.centre
    [c[1]], [c[2]]
end

function geometric_centre(nodes, nodeList) # made generic - requires Node type
    F = eltype(nodes[1].coords)
    sum = SVector{3, F}(0.0,0.0,0.0)
        for ID ∈ nodeList
            sum += nodes[ID].coords
        end
    return sum/(length(nodeList))
end

function geometry!(mesh::Mesh2)
    # centroids first: face delta, e and weight must be built from the final cell centres
    cell_faces = all_cell_faces(mesh)
    cell_centroids!(mesh, cell_faces)
    # normals point out of the owner cell, decided from each cell's topology (exact for any
    # cell shape); a test against cell centres is only the fallback for cells that do not close
    owner_signs = Mesh._owner_outward_signs_2d(mesh.cells, mesh.faces, mesh.boundaries, mesh.nodes)
    internal_face_properties!(mesh, owner_signs)
    boundary_face_properties!(mesh, owner_signs)
    cell_properties!(mesh, cell_faces)
    # correct_boundary_cell_volumes!(mesh)
    nothing
end

# Calculate face properties: area, normal, delta (internal faces)

function total_boundary_faces(mesh::Mesh2{I,F}) where {I,F}
    (; boundaries) = mesh
    nbfaces = zero(I)
    @inbounds for boundary ∈ boundaries
        nbfaces += length(boundary.facesID)
    end
    nbfaces
end

function internal_face_properties!(mesh::Mesh2{I,F}, owner_signs) where {I,F}
     (; nodes, faces, cells) = mesh
    nbfaces = total_boundary_faces(mesh)
    for facei ∈ (nbfaces + 1):length(faces) # loop over internal faces only!
        # Extract face
        face = faces[facei]

        # Node-based calculations
        (; nodesID, ownerCells) = face
        p1 = nodes[nodesID[1]]
        p2 = nodes[nodesID[2]]
        tangent = p2.coords - p1.coords
        area = norm(tangent)

        # Ownercell-based calculations
        F1 = face.centre
        C1 = cells[ownerCells[1]].centre 
        C2 = cells[ownerCells[2]].centre 

        C1F1 = F1 - C1 # distance vector from face centre to cell1 
        C2F1 = F1 - C2 # distance vector from face centre to cell2
        C1C2 = C2 - C1 # distance vector from cell1 to cell2
        
        # Calculate normal and check direction (from owner1 to owner2)
        unit_tangent = tangent/area
        normal = unit_tangent × SVector{3, F}(0, 0, 1)
        if owner_signs[facei] != 0
            normal = owner_signs[facei]*normal
        elseif C1C2⋅normal < zero(F)
            normal = -normal
        end

        # Calculate delta and interpolation weight
        weight, delta, e = Mesh.weight_delta_e(C1F1, C2F1, C1C2, normal)

        # Assign values to face
        face = @set face.area = area
        face = @set face.normal = normal
        face = @set face.delta = delta
        face = @set face.e = e
        face = @set face.weight = weight
        faces[facei] = face
    end
end

# Calculate face properties: area, normal, delta (boundary faces)
function boundary_face_properties!(mesh::Mesh2{I,F}, owner_signs) where {I,F}
    (; boundaries, nodes, faces, cells) = mesh
    for boundary ∈ boundaries
        (;facesID) = boundary
        for ID ∈ facesID
            face = faces[ID]

            # node-based calculations
            (; nodesID, ownerCells) = face
            p1 = nodes[nodesID[1]].coords
            p2 = nodes[nodesID[2]].coords
            tangent = p2 - p1
            area = norm(tangent)
            unit_tangent = tangent/area
            normal = unit_tangent × SVector{3, F}(0, 0, 1)

            # perform normal direction check
            F1 = face.centre
            C1 = cells[ownerCells[1]].centre 
            C1F1 = F1 - C1 # distance vector from face centre to cell1 

            if owner_signs[ID] != 0
                normal = owner_signs[ID]*normal
            elseif C1F1⋅normal < zero(F)
                normal = -normal
            end

            # calculate weight, delta and e 
            weight, delta, e = Mesh.weight_delta_e(C1F1, normal)

            # assign values to face
            face = @set face.area = area
            face = @set face.normal = normal
            face = @set face.delta = delta
            face = @set face.e = e
            face = @set face.weight = weight
            faces[ID] = face
        end
    end
end

# Faces of every cell, internal and boundary (corner cells have more than one boundary face)
function all_cell_faces(mesh::Mesh2)
    (; boundaries, cells) = mesh
    cell_faces = [copy(c.facesID) for c in cells]
    for boundary ∈ boundaries
        (; cellsID, facesID) = boundary
        for i ∈ eachindex(cellsID)
            push!(cell_faces[cellsID[i]], facesID[i])
        end
    end
    cell_faces
end

# Area centroid: area-weighted centroids of the triangles that fan the cell about its vertex average
function cell_centroids!(mesh::Mesh2{I,F}, cell_faces) where {I,F}
    (; nodes, faces, cells) = mesh
    for celli ∈ eachindex(cells)
        cell = cells[celli]
        c0 = cell.centre
        area = zero(F)
        moment = zero(c0)
        for fID ∈ cell_faces[celli]
            p1, p2 = (nodes[nID].coords for nID ∈ faces[fID].nodesID)
            a = norm((p1 - c0) × (p2 - c0))/2
            area += a
            moment += a*(c0 + p1 + p2)/3
        end
        area > zero(F) && (cells[celli] = @set cell.centre = moment/area)
    end
end

function cell_properties!(mesh::Mesh2{I,F}, cell_faces) where {I,F}
    (; faces, cells) = mesh
    for celli ∈ eachindex(cells)
        cell = cells[celli]
        volume = zero(F)
        empty!(cell.nsign)
        for fID ∈ cell_faces[celli]
            face = faces[fID]
            d_cf = face.centre - cell.centre
            # normals point from owner to neighbour, so they point out of the owner
            fnsign = face.ownerCells[1] == celli ? one(I) : -one(I)
            # nsign covers internal faces only, matching the internal face loops
            fID ∈ cell.facesID && push!(cell.nsign, fnsign)
            volume += (d_cf ⋅ face.normal * fnsign) * face.area
        end
        cells[celli] = @set cell.volume = F(0.5) * volume
    end
end
