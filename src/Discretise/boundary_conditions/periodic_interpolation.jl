@inline function boundary_interpolation!(
    BC::Union{PeriodicParent,Periodic}, phif::FaceScalarField, phi, 
    boundary_cellsID, time, fID)
    @inbounds begin
        (; faces) = phif.mesh
        i = fID - BC.IDs_range.start + 1
        pfID = BC.value.face_map[i] # id of periodic face
        pface = faces[pfID]
        pcID = pface.ownerCells[1]
        # pcell = cells[pcID]
        face = faces[fID]
        cID = boundary_cellsID[fID]

        # xf = faces[fID].centre
        # xC = cells[cID].centre
        # xN = cells[pcID].centre # needs translating by distance between patches!!

        delta1 = face.delta #*norm(face.e ⋅ face.normal)
        delta2 = pface.delta #*norm(pface.e ⋅ pface.normal)
        delta = delta1 + delta2
        
        # Calculate weights using normal functions
        # weight = norm(xf - xC)/norm(xN - xC)
        # weight = norm(xf - xC)/(norm(xN - xC) - BC.value.distance)
        weight = delta2/delta
        one_minus_weight = one(eltype(weight)) - weight

        # phif_values[fID] = 0.5*(phi_values[cID] + phi_values[pcID]) # linear interpolation 
        phif[fID] = weight*phi[cID] + one_minus_weight*phi[pcID]
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::Union{PeriodicParent,Periodic}, psif::FaceVectorField, psi, 
    boundary_cellsID, time, fID)
    @inbounds begin 
        (; faces) = psif.mesh
        i = fID - BC.IDs_range.start + 1
        pfID = BC.value.face_map[i] # id of periodic face
        pface = faces[pfID]
        pcID = pface.ownerCells[1]
        face = faces[fID]
        cID = boundary_cellsID[fID]

        delta1 = face.delta #*norm(face.e ⋅ face.normal)
        delta2 = pface.delta #*norm(pface.e ⋅ pface.normal)
        delta = delta1 + delta2
        w = delta2/delta
        psif[fID] = w*psi[cID] + (1.0 - w)psi[pcID] # linear interpolation 
    end
    nothing
end