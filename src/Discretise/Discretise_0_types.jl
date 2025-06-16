export AbstractScheme, AbstractBoundary
export AbstractDirichlet, AbstractNeumann, AbstractPhysicalConstraint
export KWallFunction, OmegaWallFunction, NutWallFunction
# export Constant, Linear, Upwind, LUST
export Linear, Upwind, LUST
export BoundedUpwind
export SteadyState, Euler, CrankNicolson
export Gauss, Midpoint

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
# struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end
struct LUST <: AbstractScheme end
struct BoundedUpwind <: AbstractScheme end
struct Gauss <: AbstractScheme end
struct Midpoint <: AbstractScheme end
struct SteadyState <: AbstractScheme end 
struct Euler <: AbstractScheme end 
struct CrankNicolson <: AbstractScheme end # not implemented yet


# SUPPORTED BOUNDARY CONDITIONS 

abstract type AbstractBoundary end
abstract type AbstractDirichlet <: AbstractBoundary end
abstract type AbstractNeumann <: AbstractBoundary end
abstract type AbstractWallFunction <: AbstractBoundary end
abstract type AbstractPhysicalConstraint <: AbstractBoundary end

y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

# Kwall function structure and constructor
struct KWallFunction{I,V,R<:UnitRange} <: AbstractWallFunction
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure KWallFunction
KWallFunction(name::Symbol; kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8) = begin
    yPlusLam = y_plus_laminar(E, kappa)
    KWallFunction(name, (kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam))
end
# NEED TO WRITE A GENERIC FUNCTION TO ASSIGN WALL FUNCTION BOUNDARY CONDITIONS!!!!
function fixedValue(BC::KWallFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return KWallFunction{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return KWallFunction{I,V,R<:UnitRange}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end

# Omega wall function structure and constructor
struct OmegaWallFunction{I,V,R<:UnitRange} <: AbstractWallFunction
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure OmegaWallFunction
OmegaWallFunction(name::Symbol; kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8) = begin
    yPlusLam = y_plus_laminar(E, kappa)
    OmegaWallFunction(
        name, (kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam)
        )
end

function fixedValue(BC::OmegaWallFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return OmegaWallFunction{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return OmegaWallFunction{I,V,R<:UnitRange}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end

# Nut wall function structure and constructor
struct NutWallFunction{I,V,R<:UnitRange} <: AbstractWallFunction 
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure NutWallFunction
NutWallFunction(name::Symbol; kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8) = begin
    yPlusLam = y_plus_laminar(E, kappa)
    NutWallFunction(name, (kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam))
end
function fixedValue(BC::NutWallFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return NutWallFunction{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return NutWallFunction{I,V,R<:UnitRange}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end