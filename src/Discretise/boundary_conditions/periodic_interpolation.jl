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
        pcID = faces.ownerCells[pfID][1]
        # cID = boundary_cellsID[fID]
        cID = faces.ownerCells[fID][1]


        C1 = cells.centre[cID]
        Pf = faces.centre[fID] - C1
        PN = (cells.centre[pcID] - transform.distance) - C1
        normal = faces.normal[fID]
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
        pcID = faces.ownerCells[pfID][1]
        # cID = boundary_cellsID[fID]
        cID = faces.ownerCells[fID][1]


        C1 = cells.centre[cID]
        Pf = faces.centre[fID] - C1
        PN = (cells.centre[pcID] - transform.distance) - C1
        normal = faces.normal[fID]
        wn = (Pf⋅normal)/(PN⋅normal)
        w = one(wn) - wn

        psifi = w*psi[cID] + wn*psi[pcID] # linear interpolation 
        psif[fID] = psifi
        psif[pfID] = psifi
    end
    nothing
end