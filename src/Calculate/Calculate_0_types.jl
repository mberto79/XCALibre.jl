export get_scheme
export Grad 

struct Grad{S<:AbstractScheme, I, F}
    phi::ScalarField{I,F}
    x::Vector{F}
    y::Vector{F}
    z::Vector{F}
    # phif::FaceScalarField{I,F} 
    correctors::I
    correct::Bool
    mesh::Mesh2{I,F}
end
Grad{S}(phi::ScalarField{I,F}) where {S,I,F} = begin
    mesh = phi.mesh
    (; cells) = mesh
    ncells = length(cells)
    gradx = zeros(F, ncells)
    grady = zeros(F, ncells)
    gradz = zeros(F, ncells)
    Grad{S,I,F}(phi, gradx, grady, gradz, one(I), false, mesh)
end
Grad{S}(phi::ScalarField{I,F}, correctors::I) where {S,I,F} = begin 
    mesh = phi.mesh
    (; cells) = mesh
    ncells = length(cells)
    gradx = zeros(F, ncells)
    grady = zeros(F, ncells)
    gradz = zeros(F, ncells)
    Grad{S,I,F}(phi, gradx, grady, gradz, correctors, true, mesh)
end
get_scheme(term::Grad{S,I,F}) where {S,I,F} = S
(grad::Grad{S,I,F})(i::I) where {S,I,F} = SVector{3,F}(grad.x[i], grad.y[i], grad.z[i])