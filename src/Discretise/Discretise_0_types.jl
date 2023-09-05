export AbstractScheme, AbstractBoundary
export AbstractDirichlet, AbstractNeumann
export Dirichlet, Neumann
export KWallFunction, OmegaWallFunction, NutWallFunction
export Constant, Linear, Upwind
export Steady, Euler, CrankNicolson
export Orthogonal, Midpoint
export assign, @assign!
export set_schemes

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end
struct Orthogonal <: AbstractScheme end
struct Midpoint <: AbstractScheme end
struct Steady <: AbstractScheme end 
struct Euler <: AbstractScheme end 
struct CrankNicolson <: AbstractScheme end



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
KWallFunction(name::Symbol) = begin
    KWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end

struct OmegaWallFunction{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end
OmegaWallFunction(name::Symbol) = begin
    OmegaWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end

struct NutWallFunction{I,V} <: AbstractBoundary 
    ID::I 
    value::V 
end
NutWallFunction(name::Symbol) = begin
    NutWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
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

macro assign!(model, field, BCs)
    emodel = esc(model)
    efield = Symbol(field)
    eBCs = esc(BCs)
    quote
        f = $emodel.$efield
        f = assign(f, $eBCs...)
        @reset $emodel.$efield = f
    end
end

macro assign!(model, turb, field, BCs)
    emodel = esc(model)
    eturb = Symbol(turb)
    efield = Symbol(field)
    eBCs = esc(BCs)
    quote
        f = $emodel.$eturb.$efield
        f = assign(f, $eBCs...)
        @reset $emodel.$eturb.$efield = f
    end
end

set_schemes(; 
    divergence=Linear, 
    laplacian=Linear, 
    gradient=Orthogonal) = begin
    (
        divergence=divergence,
        laplacian=laplacian,
        gradient=gradient
    )
end