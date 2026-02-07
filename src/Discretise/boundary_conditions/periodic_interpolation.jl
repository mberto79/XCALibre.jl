# Catch all interpolation for Periodic
@inline boundary_interpolation!(BC::Periodic, phif, phi, boundary_cellsID, time, fID) = 
begin
    nothing 
end

@inline function boundary_interpolation!(
    BC::PeriodicParent, phif::FaceScalarField, phi, 
    boundary_cellsID, time, fID)
    @inbounds begin
        i = fID - BC.IDs_range.start + 1
        (; transform ) = BC.value
        (; faces, cells) = phif.mesh
        pfID = BC.value.face_map[i] # id of periodic face
        pface = faces[pfID]
        pcID = pface.ownerCells[1]
        # pcell = cells[pcID]
        face = faces[fID]
        cID = boundary_cellsID[fID]

        delta1 = face.delta #*norm(face.e ⋅ face.normal)
        delta2 = pface.delta #*norm(pface.e ⋅ pface.normal)
        delta = delta1 + delta2
        weight = delta2/delta
        
        # Calculate weights using normal functions
        # xf = faces[fID].centre
        # xC = cells[cID].centre
        # xN = cells[pcID].centre - transform.distance*transform.direction
        # weight = norm(xf - xN)/norm(xN - xC)

        one_minus_weight = one(eltype(weight)) - weight

        # phif_values[fID] = 0.5*(phi_values[cID] + phi_values[pcID]) # linear interpolation
        phifi =  weight*phi[cID] + one_minus_weight*phi[pcID]
        phif[fID] = phifi
        phif[pfID] = phifi
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::PeriodicParent, psif::FaceVectorField, psi, 
    boundary_cellsID, time, fID)
    @inbounds begin 
        i = fID - BC.IDs_range.start + 1
        (; transform) = BC.value
        (; faces, cells) = psif.mesh
        pfID = BC.value.face_map[i] # id of periodic face
        pface = faces[pfID]
        pcID = pface.ownerCells[1]
        face = faces[fID]
        cID = boundary_cellsID[fID]

        delta1 = face.delta #*norm(face.e ⋅ face.normal)
        delta2 = pface.delta #*norm(pface.e ⋅ pface.normal)
        delta = delta1 + delta2
        w = delta2/delta
        one_w = one(eltype(w)) - w

        # xf = faces[fID].centre
        # xC = cells[cID].centre
        # xN = cells[pcID].centre + transform.distance*transform.direction
        # w = norm(xf - xN)/norm(xN - xC)
        # one_w = one(eltype(w)) - w

        psifi = w*psi[cID] + one_w*psi[pcID] # linear interpolation 
        psif[fID] = psifi
        psif[pfID] = psifi
    end
    nothing
end