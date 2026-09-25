@inline function boundary_interpolation!(
    BC::DirichletFunction{T,Test,R}, phif::FaceScalarField, phi, boundary_cellsID, time, fID) where {T,Test<:Function,R}
    (; faces) = phi.mesh
    @inbounds begin
        face = faces[fID]
        i = fID - BC.IDs_range.start + 1
        phif[fID] = BC.value(face.centre, time, i)
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::DirichletFunction{T,Test,R}, psif::FaceVectorField, psi, boundary_cellsID, time, fID) where {T,Test<:Function,R}
    (; faces) = psi.mesh
    @inbounds begin
        face = faces[fID]
        i = fID - BC.IDs_range.start + 1
        psif[fID] = BC.value(face.centre, time, i)
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::DirichletFunction{T,Test,R}, phif::FaceScalarField, phi, boundary_cellsID, time, fID) where {T,Test<:XCALibreUserFunctor,R}
    if BC.value.steady 
        return nothing
    end
    (; faces) = phi.mesh
    @inbounds begin
        face = faces[fID]
        i = fID - BC.IDs_range.start + 1
        phif[fID] = BC.value(face.centre, time, i)
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::DirichletFunction{T,Test,R}, psif::FaceVectorField, psi, boundary_cellsID, time, fID) where {T,Test<:XCALibreUserFunctor,R}
    if BC.value.steady
        return nothing
    end
    (; faces) = psi.mesh
    @inbounds begin
        face = faces[fID]
        i = fID - BC.IDs_range.start + 1
        psif[fID] = BC.value(face.centre, time, i)
    end
    nothing
end

