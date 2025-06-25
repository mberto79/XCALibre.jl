const ZEROGRADIENT = Union{Zerogradient, Extrapolated, KWallFunction, NutWallFunction, OmegaWallFunction}

@inline function boundary_interpolation!(
    BC::ZEROGRADIENT, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID] 
    end
    nothing
end


@inline function boundary_interpolation!(
    BC::ZEROGRADIENT, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        psif[fID] = psi[cID]
    end
    nothing
end