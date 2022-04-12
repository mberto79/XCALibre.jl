export get_scheme
export Grad 

struct Grad{S<:AbstractScheme, I, F}
    # grad::Vector{SVector{3, F}}
    x::Vector{F}
    y::Vector{F}
    z::Vector{F}
    # phif::FaceScalarField{I,F} 
    correctors::I
end
Grad{S}(mesh::Mesh2{I,F}) where {S,I,F} = begin
    (; cells) = mesh
    ncells = length(cells)
    gradx = zeros(F, ncells)
    grady = zeros(F, ncells)
    gradz = zeros(F, ncells)
    Grad{S,I,F}(gradx, grady, gradz, one(I))
end
Grad{S}(mesh::Mesh2{I,F}, correctors::I) where {S,I,F} = begin 
    (; cells) = mesh
    ncells = length(cells)
    gradx = zeros(F, ncells)
    grady = zeros(F, ncells)
    gradz = zeros(F, ncells)
    Grad{S,I,F}(gradx, grady, gradz, correctors)
end
get_scheme(term::Grad{S,I,F}) where {S,I,F} = S