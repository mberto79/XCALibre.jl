export centre2D, centre3D
export geometric_centre
export face_properties!

function centre2D(obj)
    c = obj.centre
    [c[1]], [c[2]]
end

function centre3D(obj)
    c = obj.centre
    [c[1]], [c[2]], [c[3]]
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

function face_properties!(mesh::Mesh2{I,F}) where {I,F}
    (; nodes, faces, cells) = mesh
    for facei ∈ eachindex(faces)
        # extract face
        face = faces[facei]

        # node-based calculations
        (; nodesID) = face
        p1 = nodes[nodesID[1]]
        p2 = nodes[nodesID[2]]
        tangent = p2.coords - p1.coords
        area = norm(tangent)
        unit_tangent = tangent/area
        normal = unit_tangent × UnitVectors().k
        face = @set face.area = area
        face = @set face.normal = normal

        # ownercell-based calculations
        (; ownerCells) = face
        owner1 = ownerCells[1]
        owner2 = ownerCells[2]
        c1 = cells[owner1].centre
        c2 = cells[owner2].centre
        cf = face.centre
        d_1f = cf - c1 # distance vector from cell1 to face centre
        d_f2 = c2 - cf # distance vector from face centre to cell2
        d_12 = c2 - c1 # distance vector from cell1 to cell2
        delta = abs(d_12⋅normal) # abs just in case is -ve
        weight = abs((d_1f⋅normal)/(d_1f⋅normal + d_f2⋅normal)) 
        
        # Deal with boundary faces (will move to separate function/loop later!)
        if c1 == c2
            weight = one(F)
            delta = abs(d_1f⋅normal)
        end
        face = @set face.delta = delta
        faces[facei] = @set face.weight = weight
    end
end

#= 
For cells need the following

nsign::SVector{4, I}
volume::F
=#