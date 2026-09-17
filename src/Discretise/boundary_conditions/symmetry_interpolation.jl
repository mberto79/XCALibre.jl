@inline function boundary_interpolation!(
    BC::Symmetry, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID]
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::Symmetry, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        normal = psi.mesh.faces[fID].normal
        psi_cell = psi[cID]
        psif[fID] = psi_cell - (psi_cell⋅normal)*normal
    end
    nothing
end
