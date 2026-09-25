# FixedHeatFlux face value: exact is T_f = T_c + q*delta/keff for a prescribed inward flux q,
# but keff is not reachable from boundary_interpolation! (field, boundary-cell map, face ID only),
# so a zero-gradient face value is used instead.

@inline function boundary_interpolation!(
    BC::FixedHeatFlux, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID]
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::FixedHeatFlux, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    error("FixedHeatFlux is a scalar (temperature) boundary condition and cannot \
be applied to a vector field.")
    nothing
end
