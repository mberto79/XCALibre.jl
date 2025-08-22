export AbstractField
export ScalarFloat, ConstantScalar, ConstantVector
export AbstractScalarField, ScalarField, FaceScalarField
export AbstractVectorField, VectorField, FaceVectorField
export AbstractTensorField, TensorField, T
export StrainRate, Vorticity, Dev, Sqr, MagSqr
export _mesh
export initialise!

struct ScalarFloat{DTYPE}
    zero::DTYPE 
end 

ScalarFloat(mesh::AbstractMesh) = ScalarFloat(zero(_get_float(mesh)))
@inline (scalar::ScalarFloat{DTYPE})(v::Number) where DTYPE = DTYPE(v)

# ABSTRACT TYPES

abstract type AbstractField end
abstract type AbstractScalarField <: AbstractField end
abstract type AbstractVectorField <: AbstractField end
abstract type AbstractTensorField <: AbstractField end

Base.show(io::IO, field::AbstractField) = print(io, typeof(field).name.wrapper)

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
Base.getindex(v::ConstantVector, i::Integer) = SVector{3, eltype(v.x)}(v.x, v.y, v.z)

# Handle cases where `Nothing` is passed to the constructor (e.g. in Solid constructor)
ConstantScalar(value::Nothing) = nothing
ConstantVector(value::Nothing) = nothing


# FIELDS 
"""
    struct ScalarField{VF,M,BC} <: AbstractScalarField
        values::VF  # scalar values at cell centre
        mesh::M     # reference to mesh
        BCs::BC     # store user-provided boundary conditions
    end
"""
struct ScalarField{VF,M} <: AbstractScalarField
    values::VF  # scalar values at cell centre
    mesh::M     # reference to mesh
end
Adapt.@adapt_structure ScalarField
ScalarField(mesh::AbstractMesh; store_mesh=true) =begin
    ncells  = length(mesh.cells)
    F = _get_float(mesh)
    backend = _get_backend(mesh)
    arr = KernelAbstractions.zeros(backend, F, ncells)
    if store_mesh
        return ScalarField(arr, mesh)
    else
        return ScalarField(arr, ())
    end
end

struct FaceScalarField{VF,M} <: AbstractScalarField
    values::VF
    mesh::M
end
Adapt.@adapt_structure FaceScalarField

FaceScalarField(mesh::AbstractMesh; store_mesh=true) = begin
    nfaces  = length(mesh.faces)
    F = _get_float(mesh)
    backend = _get_backend(mesh)
    arr = KernelAbstractions.zeros(backend, F, nfaces)
    if store_mesh
        return FaceScalarField(arr, mesh)
    else
        return FaceScalarField(arr, ())
    end
end

Base.getindex(s::AbstractScalarField, i::I) where I<:Integer = begin
    s.values[i]
end
Base.setindex!(s::AbstractScalarField, x, i::I) where I<:Integer = begin
    s.values[i] = x
end
Base.length(s::AbstractScalarField) = length(s.values)
Base.eachindex(s::AbstractScalarField) = eachindex(s.values)
Base.eltype(s::AbstractScalarField) = eltype(s.values)
KA.get_backend(s::AbstractScalarField) = KA.get_backend(s.values)

# VECTOR FIELD IMPLEMENTATION

"""
    struct VectorField{S1<:ScalarField,S2,S3,M<:AbstractMesh,BC} <: AbstractVectorField
        x::S1   # x-component is itself a `ScalarField`
        y::S2   # y-component is itself a `ScalarField`
        z::S3   # z-component is itself a `ScalarField`
        mesh::M
        BCs::BC
    end
"""
struct VectorField{S1<:ScalarField,S2,S3,M<:AbstractMesh} <: AbstractVectorField
    x::S1
    y::S2
    z::S3
    mesh::M
    # BCs::BC
end
Adapt.@adapt_structure VectorField

VectorField(mesh::AbstractMesh) = begin
    VectorField(
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false), 
        ScalarField(mesh, store_mesh=false), 
        mesh,
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
    FaceVectorField(
        FaceScalarField(mesh, store_mesh=false),
        FaceScalarField(mesh, store_mesh=false), 
        FaceScalarField(mesh, store_mesh=false),
        mesh)
end

Base.getindex(v::AbstractVectorField, i::Integer) = @inbounds SVector{3}(v.x[i], v.y[i], v.z[i])
Base.setindex!(v::AbstractVectorField, vec::SVector{3, T}, i::Integer) where T= begin
    # length(x) == 3 || throw("Vectors must have 3 components")
    (; x, y, z) = v
    x[i] = vec[1]
    y[i] = vec[2]
    z[i] = vec[3]
end
Base.length(v::AbstractVectorField) = length(v.x)
Base.eachindex(v::AbstractVectorField) = eachindex(v.x)
Base.eltype(v::AbstractVectorField) = eltype(v.x)
KA.get_backend(v::AbstractVectorField) = KA.get_backend(v.x)

struct Sqr{N,T<:AbstractVectorField} <: AbstractTensorField
    scale::N
    parent::T 
end
Adapt.@adapt_structure Sqr

# Sqr(scale::Number, field) = Sqr(scale, field)
Sqr(field) = Sqr(1, field)

Base.getindex(vec::Sqr{N,Field}, i::I) where {N,Field<:AbstractVectorField,I<:Integer} = begin
    vi = vec.parent[i]
    vec.scale*vi*vi'
end
_mesh(field::Sqr) = _mesh(field.parent)

struct MagSqr{N,T<:AbstractField} <: AbstractScalarField
    scale::N
    parent::T 
end
Adapt.@adapt_structure MagSqr

# MagSqr(scale::Number, field) = MagSqr(scale, field)
MagSqr(field) = MagSqr(1, field)

Base.getindex(vec::MagSqr{N,Field}, i::I) where {N,Field<:AbstractField,I<:Integer} = begin
    vi = vec.parent[i]
    vec.scale*vi⋅vi
end
_mesh(field::MagSqr) = _mesh(field.parent)

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
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false),
        ScalarField(mesh, store_mesh=false),
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

Base.setindex!(T::TensorField, t::SMatrix{3,3,F,9}, i::Integer) where F = begin
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
KA.get_backend(t::AbstractTensorField) = KA.get_backend(t.xx)
_mesh(field::AbstractField) = field.mesh # catch all accessor to mesh

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

struct Vorticity{TU, GT} <: AbstractTensorField 
    U::TU
    gradU::GT 
end
Adapt.Adapt.@adapt_structure Vorticity
_mesh(field::Vorticity) = _mesh(field.U)

Base.getindex(S::Vorticity, i::I) where {I<:Integer} = begin
    gradi = S.gradU[i]
    0.5*(gradi - gradi')
end

struct StrainRate{G, GT, TU, TUF} <: AbstractTensorField
    gradU::G
    gradUT::GT
    U::TU
    Uf::TUF
end
Adapt.@adapt_structure StrainRate
_mesh(field::StrainRate) = _mesh(field.U)

Base.getindex(S::StrainRate{G, GT, TU, TUF}, i::I) where {G, GT, TU, TUF, I<:Integer} = begin
    gradi = S.gradU[i]
    0.5*(gradi + gradi')
end

struct Dev{T<:AbstractTensorField} <: AbstractTensorField
    parent::T 
end
Adapt.@adapt_structure Dev 

Base.getindex(T::Dev{Tensor}, i::Idx) where {Tensor<:AbstractTensorField,Idx<:Integer} = begin
    Ti = T.parent[i]
    Ti - 1/3*tr(Ti)*I
end

_mesh(field::Dev) = _mesh(field.parent)

# Initialise Scalar and Vector fields
"""
    function initialise!(field, value) # dummy function for documentation
        # Assign `value` to field in-place
        nothing
    end

This function will set the given `field` to the `value` provided in-place. Useful for initialising fields prior to running a simulation.

# Input arguments

- `field` specifies the field to be initialised. The field must be either a `AbractScalarField` or `AbstractVectorField`
- `value` defines the value to be set. This should be a scalar or vector (3 components) depending on the field to be modified e.g. for an `AbstractVectorField` we can specify as `value=[10,0,0]`

Note: in most cases the fields to be modified are stored within a physics model i.e. a `Physics` object. Thus, the argument `value` must fully qualify the model. For example, if we have created a `Physics` model named `mymodel` to set the velocity field, `U`, we would set the argument `field` to `mymodel.momentum.U`. See the example below.

# Example

```julia
initialise!(mymodel.momentum.U, [2.5, 0, 0])
initialise!(mymodel.momentum.p, 1.25)
```
"""
function initialise!(field, value) # dummy function for documentation
    throw("Arguments provided for field are not of type ScalarField nor VectorField")
    nothing
end

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
    if s_type <: Number
        s.values .= convert(s_type, value)
    else
        trow("ScalarFields should be initialised with numbers. The value provided is of type $(typeof(value))")
    end
    nothing
end