export get_scheme, Grad 

struct Grad{S<:AbstractScheme, I, F}
    grad::Vector{SVector{3, F}}
    phif::FaceScalarField{I,F} 
    correctors::I
end
Grad{S}(phif::FaceScalarField{I,F}) where {S,I,F} = begin
    (; cells) = phif.mesh
    grad = [SVector{3,F}(0.0,0.0,0.0) for _ ∈ eachindex(cells)]
    Grad{S,I,F}(grad, phif, one(I))
end
Grad{S}(phif::FaceScalarField{I,F}, correctors::I) where {S,I,F} = begin 
    (; cells) = phif.mesh
    grad = [SVector{3,F}(0.0,0.0,0.0) for _ ∈ eachindex(cells)]
    Grad{S,I,F}(grad, phif, correctors)
end
get_scheme(term::Grad{S,I,F}) where {S,I,F} = S