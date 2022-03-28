export centre2d
export geometric_centre
export internal_face_properties!, boundary_face_properties!



function centre2d(face::Face2D{I,F}) where {I,F}
    c = face.centre
    [c[1]], [c[2]]
end

function centre2d(cell::Cell{I,F}) where {I,F}
    c = cell.centre
    [c[1]], [c[2]]
end

function geometric_centre(nodes::Vector{Node{F}}, nodeList::SVector{N, I}) where {I,F,N}
    sum = SVector{3, F}(0.0,0.0,0.0)
        for ID ∈ nodeList
            sum += nodes[ID].coords
        end
    return sum/(length(nodeList))
end

#= 
For faces need the following

normal::SVector{3, F}
area::F
delta::F
=#

function internal_face_properties!(mesh::Mesh2{I,F}) where {I,F}
    (; nodes, faces, cells) = mesh
    nbfaces = total_boundary_faces(mesh)
    for facei ∈ (nbfaces + 1):length(faces) # loop over internal faces only!
        # extract face
        face = faces[facei]
        # node-based calculations
        (; nodesID, ownerCells) = face
        p1 = nodes[nodesID[1]]
        p2 = nodes[nodesID[2]]
        tangent = p2.coords - p1.coords
        area = norm(tangent)
        unit_tangent = tangent/area
        normal = unit_tangent × UnitVectors().k
        face = @set face.area = area
        face = @set face.normal = normal
        # ownercell-based calculations
        c1 = cells[ownerCells[1]].centre
        c2 = cells[ownerCells[2]].centre
        cf = face.centre
        d_1f = cf - c1 # distance vector from cell1 to face centre
        d_f2 = c2 - cf # distance vector from face centre to cell2
        d_12 = c2 - c1 # distance vector from cell1 to cell2
        delta = abs(d_12⋅normal) # abs just in case is -ve
        weight = abs((d_1f⋅normal)/(d_1f⋅normal + d_f2⋅normal)) 
        face = @set face.delta = delta
        faces[facei] = @set face.weight = weight
    end
end

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
                normal = -1*normal
            end
            delta = abs(d_cf⋅normal)
            # assign values to face
            face = @set face.area = area
            face = @set face.normal = normal
            face = @set face.delta = delta
            face = @set face.weight = one(F)
            faces[ID] = face
        end
    end
end

#= 
For cells need the following

nsign::SVector{4, I}
volume::F
=#