export AbstractField, AbstractScalarField, AbstractVectorField
export ConstantScalar, ConstantVector
export ScalarField, FaceScalarField
export VectorField, FaceVectorField

# ABSTRACT TYPES

abstract type AbstractField end
abstract type AbstractScalarField <: AbstractField end
abstract type AbstractVectorField <: AbstractField end

# CONSTANT FIELDS 

struct ConstantScalar{V<:Number} <: AbstractScalarField
    values::V
end
Base.getindex(s::ConstantScalar, i::Integer) = s.values

struct ConstantVector{V<:Number} <: AbstractVectorField
    x::V
    y::V
    z::V
end
Base.getindex(v::ConstantVector, i::Integer) = SVector{3, eltype(v.x)}(v.x[i], v.y[i], v.z[i])

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

# (s::AbstractScalarField)(i::Integer) = s.values[i]
Base.getindex(s::AbstractScalarField, i::Integer) = s.values[i]
Base.setindex!(s::AbstractScalarField, x, i::Integer) = begin
    s.values[i] = x
end
Base.length(s::AbstractScalarField) = length(s.values)
Base.eachindex(s::AbstractScalarField) = eachindex(s.values)

struct VectorField{I,F} <: AbstractVectorField
    x::ScalarField{I,F}
    y::ScalarField{I,F}
    z::ScalarField{I,F}
    mesh::Mesh2{I,F}
end
VectorField(mesh::Mesh2{I,F}) where {I,F} = begin
    ncells = length(mesh.cells)
    VectorField(
        ScalarField(zeros(F, ncells), mesh),
        ScalarField(zeros(F, ncells), mesh), 
        ScalarField(zeros(F, ncells), mesh), 
        mesh)
end

struct FaceVectorField{I,F} <: AbstractVectorField
    x::FaceScalarField{I,F}
    y::FaceScalarField{I,F}
    z::FaceScalarField{I,F}
    mesh::Mesh2{I,F}
end
FaceVectorField(mesh::Mesh2{I,F}) where {I,F} = begin
    nfaces = length(mesh.faces)
    FaceVectorField(
        FaceScalarField(zeros(F, nfaces), mesh),
        FaceScalarField(zeros(F, nfaces), mesh), 
        FaceScalarField(zeros(F, nfaces), mesh),
        mesh)
end

Base.getindex(v::AbstractVectorField, i::Integer) = SVector{3, eltype(v.x)}(v.x[i], v.y[i], v.z[i])
Base.setindex!(v::AbstractVectorField, x::AbstractVector, i::Integer) = begin
    length(x) == 3 || throw("Vectors must have 3 components")
    v.x[i] = x[1]
    v.y[i] = y[2]
    v.z[i] = z[3]
end

# function initialise!(v::AbstractVectorField, vec::Vector{T}) where T
#     n = length(vec)
#     if T !== eltype(v.x)
#         throw("Vectors are not the same type: $(eltype(v.x)) is not $T")
#     elseif n == 3
#         v.x .= vec[1]
#         v.y .= vec[2]
#         v.z .= vec[3]
#     else
#         throw("Vectors should have 3 components")
#     end
#     nothing
# end