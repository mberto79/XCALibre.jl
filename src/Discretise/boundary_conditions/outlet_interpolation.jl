@inline function boundary_interpolation!(
    BC::Outlet, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID]
    end
    nothing
end


@inline function boundary_interpolation!(
    BC::Outlet, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        psif[fID] = psi[cID]
    end
    nothing
end
