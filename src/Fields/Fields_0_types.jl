export AbstractField
export ConstantScalar, ConstantVector
export AbstractScalarField, ScalarField, FaceScalarField
export AbstractVectorField, VectorField, FaceVectorField
export AbstractTensorField, TensorField
export initialise!

# ABSTRACT TYPES

abstract type AbstractField end
abstract type AbstractScalarField <: AbstractField end
abstract type AbstractVectorField <: AbstractField end
abstract type AbstractTensorField <: AbstractField end

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

struct ScalarField{F,M<:Mesh2,BC} <: AbstractScalarField
    values::Vector{F}
    mesh::M
    BCs::BC
end
ScalarField(mesh::Mesh2) =begin
    ncells  = length(mesh.cells)
    F = eltype(mesh.nodes[1].coords)
    ScalarField(zeros(F,ncells), mesh, ())
end

struct FaceScalarField{F,M<:Mesh2} <: AbstractScalarField
    values::Vector{F}
    mesh::M
end
FaceScalarField(mesh::Mesh2) =begin
    nfaces  = length(mesh.faces)
    F = eltype(mesh.nodes[1].coords)
    FaceScalarField(zeros(F,nfaces), mesh)
end

# (s::AbstractScalarField)(i::Integer) = s.values[i]
Base.getindex(s::AbstractScalarField, i::I) where I<:Integer = begin
    s.values[i]
end
Base.setindex!(s::AbstractScalarField, x, i::I) where I<:Integer = begin
    s.values[i] = x
end
Base.length(s::AbstractScalarField) = length(s.values)
Base.eachindex(s::AbstractScalarField) = eachindex(s.values)

struct VectorField{S1<:ScalarField,S2,S3,M<:Mesh2,BC} <: AbstractVectorField
    x::S1
    y::S2
    z::S3
    mesh::M
    BCs::BC
end
VectorField(mesh::Mesh2) = begin
    ncells = length(mesh.cells)
    F = eltype(mesh.nodes[1].coords)
    VectorField(
        ScalarField(zeros(F, ncells), mesh, ()),
        ScalarField(zeros(F, ncells), mesh, ()), 
        ScalarField(zeros(F, ncells), mesh, ()), 
        mesh,
        () # to hold x, y, z and combined BCs
        )
end

struct FaceVectorField{S1<:FaceScalarField,S2,S3,M} <: AbstractVectorField
    x::S1
    y::S2
    z::S3
    mesh::M
end
FaceVectorField(mesh::Mesh2) = begin
    nfaces = length(mesh.faces)
    F = eltype(mesh.nodes[1].coords)
    FaceVectorField(
        FaceScalarField(zeros(F, nfaces), mesh),
        FaceScalarField(zeros(F, nfaces), mesh), 
        FaceScalarField(zeros(F, nfaces), mesh),
        mesh)
end

Base.getindex(v::AbstractVectorField, i::Integer) = SVector{3, eltype(v.x)}(v.x[i], v.y[i], v.z[i])
Base.setindex!(v::AbstractVectorField, x::SVector{3, T}, i::Integer) where T= begin
    # length(x) == 3 || throw("Vectors must have 3 components")
    v.x[i] = x[1]
    v.y[i] = y[2]
    v.z[i] = z[3]
end

struct TensorField{S1,S2,S3,S4,S5,S6,S7,S8,S9,M} <: AbstractTensorField
    xx::S1
    xy::S2
    xz::S3 
    yx::S4 
    yy::S5 
    yz::S6 
    zx::S7 
    zy::S8
    zz::S9
    mesh::M
end

TensorField(mesh::Mesh2) = begin
    TensorField(
        ScalarField(mesh),
        ScalarField(mesh),
        ScalarField(mesh),
        ScalarField(mesh),
        ScalarField(mesh),
        ScalarField(mesh),
        ScalarField(mesh),
        ScalarField(mesh),
        ScalarField(mesh),
        mesh
    )
end

Base.getindex(T::TensorField, i::Integer) = begin
    Tf = eltype(T.xx.values)
    SMatrix{3,3,Tf,9}(
        T.xx[i],
        T.yx[i],
        T.zx[i],
        T.xy[i],
        T.yy[i],
        T.zy[i],
        T.xz[i],
        T.yz[i],
        T.zz[i],
        )
end

# struct Transpose{T<:Grad}
#     parent::T
# end
# Base.getindex(t::Transpose{Grad{S,F,X,Y,Z,I,M}}, i::Integer) where {S,F,X<:Grad,Y,Z,I,M} = begin
#     gradt = t.parent
#     Tf = eltype(gradt.x.x)
#     SMatrix{3,3,Tf,9}(
#         gradt.x.x[i],
#         gradt.y.x[i],
#         gradt.z.x[i],
#         gradt.x.y[i],
#         gradt.y.y[i],
#         gradt.z.y[i],
#         gradt.x.z[i],
#         gradt.y.z[i],
#         gradt.z.z[i],
#         )
# end

# Initialise Scalar and Vector fields

function initialise!(v::AbstractVectorField, vec::Vector{T}) where T
    n = length(vec)
    v_type = eltype(v.x.values)
    if n == 3
        v.x.values .= convert(v_type, vec[1])
        v.y.values .= convert(v_type, vec[2])
        v.z.values .= convert(v_type, vec[3])
    else
        throw("Vectors should have 3 components")
    end
    nothing
end

function initialise!(s::AbstractScalarField, value::V) where V
    s_type = eltype(s.values)
    s.values .= convert(s_type, value)
    nothing
end