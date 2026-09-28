@inline function boundary_interpolation!(
    BC::Dirichlet, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds phif[fID] = BC.value
    nothing
end

@inline function boundary_interpolation!(
    BC::Dirichlet, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds psif[fID] = BC.value
    nothing
end
