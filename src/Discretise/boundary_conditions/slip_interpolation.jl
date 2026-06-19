@inline function boundary_interpolation!(
    BC::Slip, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID] 
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::Slip, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        face = psi.mesh.faces[fID]
        (; normal) = face

        psi_cell = psi[cID]
        
        psi_cell = psi[cID]
        psi_normal = (psi_cell⋅normal)*normal

        psif[fID] = psi_cell - psi_normal
    end
    nothing
end