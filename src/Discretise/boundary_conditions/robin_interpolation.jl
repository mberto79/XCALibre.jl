@inline function boundary_interpolation!(
    BC::Robin, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        (; faces) = phi.mesh
        face = faces[fID]
        (; delta) = face
        cID = boundary_cellsID[fID]
        (; a, b, value) = BC.value
        phif[fID] = (value*delta + b*phi[cID]) / (a*delta + b)
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::Robin, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds begin
        error("Robin boundary condition for vector fields is not implemented yet.")
    end
    nothing
end
