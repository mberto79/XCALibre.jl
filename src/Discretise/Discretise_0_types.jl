export AbstractScheme, AbstractBoundary
export AbstractDirichlet, AbstractNeumann
export Dirichlet, fixedValue, Neumann
export FixedTemperature
export Wall, Symmetry
export KWallFunction, OmegaWallFunction, NutWallFunction
export Constant, Linear, Upwind, LUST
export BoundedUpwind
export SteadyState, Euler, CrankNicolson
export Orthogonal, Midpoint
export set_schemes

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end
struct LUST <: AbstractScheme end
struct BoundedUpwind <: AbstractScheme end
struct Orthogonal <: AbstractScheme end
struct Midpoint <: AbstractScheme end
struct SteadyState <: AbstractScheme end 
struct Euler <: AbstractScheme end 
struct CrankNicolson <: AbstractScheme end # not implemented yet



# SUPPORTED BOUNDARY CONDITIONS 

abstract type AbstractBoundary end
abstract type AbstractDirichlet <: AbstractBoundary end
abstract type AbstractNeumann <: AbstractBoundary end
abstract type AbstractWallFunction <: AbstractDirichlet end

# Dirichlet structure and constructor function
struct Dirichlet{I,V} <: AbstractDirichlet
    ID::I
    value::V
end
Adapt.@adapt_structure Dirichlet
function fixedValue(BC::AbstractDirichlet, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Dirichlet{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Dirichlet{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

# Neumann structure and constructor function
struct Neumann{I,V} <: AbstractNeumann
    ID::I 
    value::V 
end
Adapt.@adapt_structure Neumann
function fixedValue(BC::AbstractNeumann, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: value is scalar
    if V <: Number
        return Neumann{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Neumann{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid        
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

# FixedTemperature Boundary condition (temporary approach for now)
struct FixedTemperature{I,V} <: AbstractDirichlet
    ID::I 
    value::V 
end
Adapt.@adapt_structure FixedTemperature
FixedTemperature(name; T, model) = begin
    FixedTemperature(name, (; T=T, energy_model=model))
end

function fixedValue(BC::FixedTemperature, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return FixedTemperature{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return FixedTemperature{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end

struct Wall{I,V} <: AbstractDirichlet
    ID::I
    value::V
end
Adapt.@adapt_structure Wall
function fixedValue(BC::Wall, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Wall{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Wall{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

struct Symmetry{I,V} <: AbstractNeumann
    ID::I
    value::V
end
Adapt.@adapt_structure Symmetry
function fixedValue(BC::Symmetry, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Symmetry{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Symmetry{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

# Kwall function structure and constructor
struct KWallFunction{I,V} <: AbstractWallFunction
    ID::I 
    value::V 
end
Adapt.@adapt_structure KWallFunction
KWallFunction(name::Symbol) = begin
    KWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end
# NEED TO WRITE A GENERIC FUNCTION TO ASSIGN WALL FUNCTION BOUNDARY CONDITIONS!!!!
function fixedValue(BC::KWallFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return KWallFunction{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return KWallFunction{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end

# Omega wall function structure and constructor
struct OmegaWallFunction{I,V} <: AbstractWallFunction
    ID::I 
    value::V 
end
Adapt.@adapt_structure OmegaWallFunction
OmegaWallFunction(name::Symbol) = begin
    OmegaWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end
function fixedValue(BC::OmegaWallFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return OmegaWallFunction{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return OmegaWallFunction{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end

# Nut wall function structure and constructor
struct NutWallFunction{I,V} <: AbstractWallFunction 
    ID::I 
    value::V 
end
Adapt.@adapt_structure NutWallFunction
NutWallFunction(name::Symbol) = begin
    NutWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end
function fixedValue(BC::NutWallFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return NutWallFunction{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return NutWallFunction{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end

