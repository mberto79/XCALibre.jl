
@inline function boundary_interpolation!(
    BC::RotatingWall, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    mesh = psi.mesh 
    faces = mesh.faces
    @inbounds psif[fID] = BC.value(faces[fID])
    nothing
end