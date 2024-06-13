export AbstractField
export ConstantScalar, ConstantVector
export AbstractScalarField, ScalarField, FaceScalarField
export AbstractVectorField, VectorField, FaceVectorField
export AbstractTensorField, TensorField, T
export initialise!

# ABSTRACT TYPES

abstract type AbstractField end
abstract type AbstractScalarField <: AbstractField end
abstract type AbstractVectorField <: AbstractField end
abstract type AbstractTensorField <: AbstractField end

Base.getindex(field::AbstractField, i::I) where I<:Integer = begin
    field(i)
end

# CONSTANT FIELDS 

struct ConstantScalar{V<:Number} <: AbstractScalarField
    values::V
end
Adapt.@adapt_structure ConstantScalar
Base.getindex(s::ConstantScalar, i::Integer) = s.values

struct ConstantVector{V<:Number} <: AbstractVectorField
    x::V
    y::V
    z::V
end
Adapt.@adapt_structure ConstantVector
Base.getindex(v::ConstantVector, i::Integer) = SVector{3, eltype(v.x)}(v.x[i], v.y[i], v.z[i])

# FIELDS 

struct ScalarField{VF,M<:AbstractMesh,BC} <: AbstractScalarField
    values::VF#Vector{F}
    mesh::M
    BCs::BC
end
Adapt.@adapt_structure ScalarField
ScalarField(mesh::AbstractMesh) =begin
    ncells  = length(mesh.cells)
    F = _get_float(mesh)
    backend = _get_backend(mesh)
    arr = _convert_array!(zeros(F,ncells), backend)
    ScalarField(arr, mesh, ())
end

struct FaceScalarField{VF,M<:AbstractMesh} <: AbstractScalarField
    values::VF#Vector{F}
    mesh::M
end
Adapt.@adapt_structure FaceScalarField
FaceScalarField(mesh::AbstractMesh) = begin
    nfaces  = length(mesh.faces)
    F = _get_float(mesh)
    backend = _get_backend(mesh)
    arr = _convert_array!(zeros(F,nfaces), backend)
    FaceScalarField(arr, mesh) #Make it pretty
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

# VECTOR FIELD IMPLEMENTATION

struct VectorField{S1<:ScalarField,S2,S3,M<:AbstractMesh,BC} <: AbstractVectorField
    x::S1
    y::S2
    z::S3
    mesh::M
    BCs::BC
end
Adapt.@adapt_structure VectorField
VectorField(mesh::AbstractMesh) = begin
    ncells = length(mesh.cells)
    F = _get_float(mesh) #eltype(mesh.nodes[1].coords) #TEMPORARY SOLUTION, RUN BY HUMBERTO
    backend = _get_backend(mesh)
    arr1 = _convert_array!(zeros(F,ncells), backend)
    arr2 = _convert_array!(zeros(F,ncells), backend)
    arr3 = _convert_array!(zeros(F,ncells), backend)
    VectorField(
        ScalarField(arr1, mesh, ()),
        ScalarField(arr2, mesh, ()), 
        ScalarField(arr3, mesh, ()), 
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
Adapt.@adapt_structure FaceVectorField
FaceVectorField(mesh::AbstractMesh) = begin
    nfaces = length(mesh.faces)
    F = _get_float(mesh)
    backend = _get_backend(mesh)
    arr1 = _convert_array!(zeros(F,nfaces), backend)
    arr2 = _convert_array!(zeros(F,nfaces), backend)
    arr3 = _convert_array!(zeros(F,nfaces), backend)
    FaceVectorField(
        FaceScalarField(arr1, mesh),
        FaceScalarField(arr2, mesh), 
        FaceScalarField(arr3, mesh),
        mesh)
end

# Base.getindex(v::AbstractVectorField, i::Integer) = @inbounds SVector{3, eltype(v.x)}(v.x[i], v.y[i], v.z[i])
Base.getindex(v::AbstractVectorField, i::Integer) = @inbounds SVector{3}(v.x[i], v.y[i], v.z[i])
Base.setindex!(v::AbstractVectorField, x::SVector{3, T}, i::Integer) where T= begin
    # length(x) == 3 || throw("Vectors must have 3 components")
    v.x[i] = x[1]
    v.y[i] = y[2]
    v.z[i] = z[3]
end
Base.length(v::AbstractVectorField) = length(v.x)
Base.eachindex(v::AbstractVectorField) = eachindex(v.x)

# TENSORFIELD IMPLEMENTATION

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
Adapt.@adapt_structure TensorField
TensorField(mesh::AbstractMesh) = begin
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

Base.setindex!(T::TensorField, t::SMatrix{3,3,F,9}, i::Integer) where F= begin
    T.xx[i] = t[1,1]
    T.yx[i] = t[2,1]
    T.zx[i] = t[3,1]
    T.xy[i] = t[1,2]
    T.yy[i] = t[2,2]
    T.zy[i] = t[3,2]
    T.xz[i] = t[1,3]
    T.yz[i] = t[2,3]
    T.zz[i] = t[3,3]
end

Base.length(t::AbstractTensorField) = length(t.xx)
Base.eachindex(t::AbstractTensorField) = eachindex(t.xx)

# TRANSPOSE IMPLEMENTATION

struct T{F<:AbstractField} # Needs to be abstractTensor type
    parent::F
end
Adapt.@adapt_structure T
Base.getindex(t::T{F}, i::Integer) where F<:TensorField = begin # type calls need sorting
    T = t.parent
    Tf = eltype(T.xx.values)
    SMatrix{3,3,Tf,9}(
        T.xx[i],
        T.xy[i],
        T.xz[i],
        T.yx[i],
        T.yy[i],
        T.yz[i],
        T.zx[i],
        T.zy[i],
        T.zz[i],
        )
end

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