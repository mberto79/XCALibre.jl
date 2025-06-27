export AbstractField
export ConstantScalar, ConstantVector
export AbstractScalarField, ScalarField, FaceScalarField
export AbstractVectorField, VectorField, FaceVectorField
export AbstractTensorField, TensorField, T
export StrainRate
export initialise!

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

# FIELDS 
"""
    struct ScalarField{VF,M<:AbstractMesh,BC} <: AbstractScalarField
        values::VF  # scalar values at cell centre
        mesh::M     # reference to mesh
        BCs::BC     # store user-provided boundary conditions
    end
"""
struct ScalarField{VF,M<:AbstractMesh} <: AbstractScalarField
    values::VF  # scalar values at cell centre
    mesh::M     # reference to mesh
    # BCs::BC     # store user-provided boundary conditions
end
Adapt.@adapt_structure ScalarField
ScalarField(mesh::AbstractMesh) =begin
    ncells  = length(mesh.cells)
    F = _get_float(mesh)
    backend = _get_backend(mesh)
    # arr = _convert_array!(zeros(F,ncells), backend)
    arr = KernelAbstractions.zeros(backend, F, ncells)
    # ScalarField(arr, mesh, ())
    ScalarField(arr, mesh)
end
# ScalarField(values::Vector{Float64}, mesh::AbstractMesh) =begin
#     ncells  = length(mesh.cells)
#     F = _get_float(mesh)
#     backend = _get_backend(mesh)
#     arr = _convert_array!(values, backend)
#     # ScalarField(arr, mesh, ())
#     ScalarField(arr, mesh)
# end

struct FaceScalarField{VF,M<:AbstractMesh} <: AbstractScalarField
    values::VF#Vector{F}
    mesh::M
end
Adapt.@adapt_structure FaceScalarField
FaceScalarField(mesh::AbstractMesh) = begin
    nfaces  = length(mesh.faces)
    F = _get_float(mesh)
    backend = _get_backend(mesh)
    # arr = _convert_array!(zeros(F,nfaces), backend)
    arr = KernelAbstractions.zeros(backend, F, nfaces)
    FaceScalarField(arr, mesh) #Make it pretty
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
    ncells = length(mesh.cells)
    F = _get_float(mesh) #eltype(mesh.nodes[1].coords) #TEMPORARY SOLUTION, RUN BY HUMBERTO
    backend = _get_backend(mesh)
    # arr1 = _convert_array!(zeros(F,ncells), backend)
    # arr2 = _convert_array!(zeros(F,ncells), backend)
    # arr3 = _convert_array!(zeros(F,ncells), backend)

    arr1 = KernelAbstractions.zeros(backend, F, ncells)
    arr2 = KernelAbstractions.zeros(backend, F, ncells)
    arr3 = KernelAbstractions.zeros(backend, F, ncells)

    
    VectorField(
        # ScalarField(arr1, mesh, ()),
        # ScalarField(arr2, mesh, ()), 
        # ScalarField(arr3, mesh, ()), 
        ScalarField(arr1, mesh),
        ScalarField(arr2, mesh), 
        ScalarField(arr3, mesh), 
        mesh,
        # () # to hold x, y, z and combined BCs
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
    # arr1 = _convert_array!(zeros(F,nfaces), backend)
    # arr2 = _convert_array!(zeros(F,nfaces), backend)
    # arr3 = _convert_array!(zeros(F,nfaces), backend)

    arr1 = KernelAbstractions.zeros(backend, F, nfaces)
    arr2 = KernelAbstractions.zeros(backend, F, nfaces)
    arr3 = KernelAbstractions.zeros(backend, F, nfaces)

    
    FaceVectorField(
        FaceScalarField(arr1, mesh),
        FaceScalarField(arr2, mesh), 
        FaceScalarField(arr3, mesh),
        mesh)
end

# Base.getindex(v::AbstractVectorField, i::Integer) = @inbounds SVector{3, eltype(v.x)}(v.x[i], v.y[i], v.z[i])
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

struct StrainRate{G, GT, TU, TUF} <: AbstractTensorField
    gradU::G
    gradUT::GT
    U::TU
    Uf::TUF
end
Adapt.@adapt_structure StrainRate

Base.getindex(S::StrainRate{G,GT}, i::I) where {G,GT,I<:Integer} = begin
    0.5.*(S.gradU[i] .+ S.gradUT[i])
end

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