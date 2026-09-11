# Face-value interpolation for `FixedHeatFlux`.
#
# The exact face temperature for a prescribed inward flux `q` is
#
#     T_f = T_c + q*delta/keff
#
# but `keff` is not reachable from this interface (`boundary_interpolation!`
# receives only the field, the boundary-cell map and the face ID). A
# zero-gradient face value is used instead.
#
# This does NOT weaken the boundary condition itself: the prescribed flux enters
# the matrix exactly, as a known source with no diagonal contribution (see
# fixedHeatFlux.jl), so the imposed wall heat load is exact regardless of the
# interpolated face value. The face value only affects auxiliary quantities such
# as gradient reconstruction of the temperature field near the wall.

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
