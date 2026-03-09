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
        pcell = cells[pcID]
        face = faces[fID]
        # cID = boundary_cellsID[fID]
        cID = face.ownerCells[1]
        cell = cells[cID]

        # delta1 = face.delta #*norm(face.e ⋅ face.normal)
        # delta2 = pface.delta #*norm(pface.e ⋅ pface.normal)
        # delta = delta1 + delta2
        # w = delta2/delta

        Pf = face.centre - cell.centre
        PN = (pcell.centre - transform.distance) - cell.centre
        normal = face.normal
        wn = (Pf⋅normal)/(PN⋅normal)
        w = one(wn) - wn

        phifi =  w*phi[cID] + wn*phi[pcID]
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
        (; transform ) = BC.value
        (; faces, cells) = psif.mesh
        pfID = BC.value.face_map[i] # id of periodic face
        pface = faces[pfID]
        pcID = pface.ownerCells[1]
        pcell = cells[pcID]
        face = faces[fID]
        # cID = boundary_cellsID[fID]
        cID = face.ownerCells[1]
        cell = cells[cID]

        # delta1 = face.delta #*norm(face.e ⋅ face.normal)
        # delta2 = pface.delta #*norm(pface.e ⋅ pface.normal)
        # delta = delta1 + delta2
        # w = delta2/delta

        Pf = face.centre - cell.centre
        PN = (pcell.centre - transform.distance) - cell.centre
        normal = face.normal
        wn = (Pf⋅normal)/(PN⋅normal)
        w = one(wn) - wn

        psifi = w*psi[cID] + wn*psi[pcID] # linear interpolation 
        psif[fID] = psifi
        psif[pfID] = psifi
    end
    nothing
end