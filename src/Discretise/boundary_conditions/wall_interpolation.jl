@inline function boundary_interpolation!(
    BC::Wall, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID] 
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::Wall, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds psif[fID] = BC.value
    nothing
end


