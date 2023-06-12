export AbstractField, AbstractScalarField, AbstractVectorField
export ScalarField, FaceScalarField
export VectorField, FaceVectorField

# ABSTRACT TYPES

abstract type AbstractField end
abstract type AbstractScalarField <: AbstractField end
abstract type AbstractVectorField <: AbstractField end

# FIELDS 

struct ScalarField{I,F} <: AbstractScalarField
    values::Vector{F}
    mesh::Mesh2{I,F}
end
ScalarField(mesh::Mesh2{I,F}) where {I,F} =begin
    ncells  = length(mesh.cells)
    ScalarField(zeros(F,ncells), mesh)
end

struct FaceScalarField{I,F} <: AbstractScalarField
    values::Vector{F}
    mesh::Mesh2{I,F}
end
FaceScalarField(mesh::Mesh2{I,F}) where {I,F} =begin
    nfaces  = length(mesh.faces)
    FaceScalarField(zeros(F,nfaces), mesh)
end

(s::AbstractScalarField)(i::Integer) = s.values[i]

struct VectorField{I,F} <: AbstractVectorField
    x::Vector{F}
    y::Vector{F}
    z::Vector{F}
    mesh::Mesh2{I,F}
end
VectorField(mesh::Mesh2{I,F}) where {I,F} = begin
    ncells = length(mesh.cells)
    VectorField(zeros(F, ncells), zeros(F, ncells), zeros(F, ncells), mesh)
end

struct FaceVectorField{I,F} <: AbstractVectorField
    x::Vector{F}
    y::Vector{F}
    z::Vector{F}
    mesh::Mesh2{I,F}
end
FaceVectorField(mesh::Mesh2{I,F}) where {I,F} = begin
    nfaces = length(mesh.faces)
    FaceVectorField(zeros(F, nfaces), zeros(F, nfaces), zeros(F, nfaces), mesh)
end

(v::AbstractVectorField)(i::Integer) = SVector{3, eltype(v.x)}(v.x[i], v.y[i], v.z[i])

function initialise!(v::AbstractVectorField, vec::Vector{T}) where T
    n = length(vec)
    if T !== eltype(v.x)
        throw("Vectors are not the same type: $(eltype(v.x)) is not $T")
    elseif n == 3
        v.x .= vec[1]
        v.y .= vec[2]
        v.z .= vec[3]
    else
        throw("Vectors should have 3 components")
    end
    nothing
end