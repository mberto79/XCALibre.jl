
@inline function boundary_interpolation!(
    BC::Neumann, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        (; faces) = phi.mesh
        delta = faces.delta[fID]
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID] + delta*BC.value 
    end
    nothing
end


@inline function boundary_interpolation!(
    BC::Neumann, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds begin
        error("Neumann boundary condition for vector fields is not implemented yet.")
    end
    nothing
end
