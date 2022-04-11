export interpolate!

function interpolate!(::Type{Linear}, phif, phi)
    start = total_boundary_faces(phi.mesh) + 1
    (; mesh, values) = phi
    (; cells, faces) = mesh
    for fi ∈ start:length(faces)
        (; ownerCells, centre, area, normal) = faces[fi]
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        c1 = cells[cID1].centre
        c2 = cells[cID2].centre
        c1_f = centre - c1
        c1_c2 = c2 - c1
        q = (c1_f⋅c1_c2)/(c1_c2⋅c1_c2)
        f_prime = c1 - q*(c1 - c2)
        w = norm( (c2 - f_prime)/(c2 - c1) )
        phi1 = values[cID1]
        phi2 = values[cID2]
        phif.values[fi] = w*phi1 + (1.0 - w)*phi2
    end
end