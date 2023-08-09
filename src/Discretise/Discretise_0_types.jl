export AbstractScheme, Constant, Linear, Upwind, Midpoint
export AbstractBoundary, AbstractDirichlet, AbstractNeumann
export Dirichlet, Neumann, KWallFunction, OmegaWallFunction 
export assign

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end
struct Midpoint <: AbstractScheme end



# SUPPORTED BOUNDARY CONDITIONS 

abstract type AbstractBoundary end
abstract type AbstractDirichlet <: AbstractBoundary end
abstract type AbstractNeumann <: AbstractBoundary end

struct Dirichlet{I,V} <: AbstractBoundary
    ID::I
    value::V
end

function Dirichlet(ID::I, value::V) where {I<:Integer,V}
    if V <: Number
        return Dirichlet{I,eltype(value)}(ID, value)
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Dirichlet{I,typeof(nvalue)}(ID, nvalue)
        else
            throw("Only vectors with three components can be used")
        end
    else
        throw("The value provided should be a scalar or a vector")
    end
end

struct Neumann{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end

struct KWallFunction{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end

struct OmegaWallFunction{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end

assign(vec::VectorField, args...) = begin
    boundaries = vec.mesh.boundaries
    @reset vec.x.BCs = ()
    @reset vec.y.BCs = ()
    @reset vec.z.BCs = ()
    @reset vec.BCs = ()
    for arg ∈ args
        bc_type = Base.typename(typeof(arg)).wrapper
        idx = boundary_index(boundaries, arg.ID)
        println("calling abstraction: ", idx)
        if typeof(arg.value) <: AbstractVector
            length(arg.value) == 3 || throw("Vector must have 3 components")
            xBCs = (vec.x.BCs..., bc_type(idx, arg.value[1]))
            yBCs = (vec.y.BCs..., bc_type(idx, arg.value[2]))
            zBCs = (vec.z.BCs..., bc_type(idx, arg.value[3]))
            uBCs = (vec.BCs..., bc_type(idx, arg.value))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        else
            xBCs = (vec.x.BCs..., bc_type(idx, arg.value))
            yBCs = (vec.y.BCs..., bc_type(idx, arg.value))
            zBCs = (vec.z.BCs..., bc_type(idx, arg.value))
            uBCs = (vec.BCs..., bc_type(idx, arg.value))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        end
    end
    return vec
end

assign(scalar::ScalarField, args...) = begin
    boundaries = scalar.mesh.boundaries
    @reset scalar.BCs = ()
    for arg ∈ args
        bc_type = Base.typename(typeof(arg)).wrapper
        idx = boundary_index(boundaries, arg.ID)
        println("calling abstraction: ", idx)
        BCs = (scalar.BCs..., bc_type(idx, arg.value))
        @reset scalar.BCs = BCs
    end
    return scalar
end