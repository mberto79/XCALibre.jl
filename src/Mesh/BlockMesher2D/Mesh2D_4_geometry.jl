export geometry!
export centre2d
export geometric_centre

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

function geometry!(mesh::Mesh2{I,F}) where {I,F}
    internal_face_properties!(mesh)
    boundary_face_properties!(mesh)
    cell_properties!(mesh)
    correct_boundary_cell_volumes!(mesh)
    nothing
end

# Calculate face properties: area, normal, delta (internal faces)
function internal_face_properties!(mesh::Mesh2{I,F}) where {I,F}
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
        c1 = cells[ownerCells[1]].centre
        c2 = cells[ownerCells[2]].centre
        cf = face.centre
        d_1f = cf - c1 # distance vector from cell1 to face centre
        d_f2 = c2 - cf # distance vector from face centre to cell2
        d_12 = c2 - c1 # distance vector from cell1 to cell2

        # Calculate normal and check direction (from owner1 to owner2)
        unit_tangent = tangent/area
        normal = unit_tangent × UnitVectors().k
        if d_12⋅normal < zero(F)
            normal = -1.0*normal
        end

        # Calculate delta and interpolation weight
        delta = norm(d_12) 
        e = d_12/delta
        weight = norm(d_f2)/norm(d_12⋅normal)

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
function boundary_face_properties!(mesh::Mesh2{I,F}) where {I,F}
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
            normal = unit_tangent × UnitVectors().k

            # perform normal direction check
            cf = face.centre
            cc = cells[ownerCells[1]].centre
            d_cf = cf - cc # distance vector from cell to face centre
            if d_cf⋅normal < zero(F)
                normal = -1.0*normal
            end
            # delta = abs(d_cf⋅normal) # face-normal distance
            delta = norm(d_cf) # exact distance
            e = d_cf/delta

            # assign values to face
            face = @set face.area = area
            face = @set face.normal = normal
            face = @set face.delta = delta
            face = @set face.e = e
            face = @set face.weight = one(F)
            faces[ID] = face
        end
    end
end

function cell_properties!(mesh::Mesh2{I,F}) where {I,F}
    (; nodes, faces, cells) = mesh
    for celli ∈ eachindex(cells)
        cell = cells[celli]
        (; centre, nsign, facesID) = cell
        volume = zero(F)
        # loop over faces: check normals and calculate volume
        for i ∈ eachindex(facesID)
            ID = facesID[i]
            face = faces[ID]
            fcentre = face.centre
            fnormal = face.normal
            farea = face.area
            d_cf = fcentre - centre # x_f : face location from cell centre
            fnsign = zero(I)
            if d_cf⋅fnormal > zero(F) # normal direction check
                fnsign = one(I)
            else
                fnsign = -one(I)
            end
            push!(nsign, fnsign)
            volume += (d_cf ⋅ fnormal*fnsign)*farea
            cells[celli] = @set cell.volume = 0.5*volume
        end
    end
end

function correct_boundary_cell_volumes!(mesh::Mesh2{I,F}) where {I,F}
    (; boundaries, faces, cells) = mesh
    for boundary ∈ boundaries
        (; cellsID, facesID) = boundary
        for i ∈ eachindex(cellsID)
            cell = cells[cellsID[i]]
            centre = cell.centre
            face = faces[facesID[i]]
            fcentre = face.centre
            fnormal = face.normal
            farea = face.area
            d_cf = fcentre - centre
            volume = cell.volume + 0.5*(d_cf ⋅ fnormal)*farea
            cells[cellsID[i]] = @set cell.volume = volume
        end
    end
end