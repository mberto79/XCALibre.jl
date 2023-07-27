export AbstractOperator, AbstractSource   
export Operator, Source, Src
export Laplacian, Divergence
export Model 

# ABSTRACT TYPES 

abstract type AbstractSource end
abstract type AbstractOperator end

# OPERATORS

# Base Operator

struct Operator{F,P,S,T}
    flux::F
    phi::P 
    sign::S
    type::T
end

# operators

struct Laplacian{T} <: AbstractOperator end
struct Divergence{T} <: AbstractOperator end

# constructors

Laplacian{T}(flux, phi) where T = Operator(
    flux, phi, 1, Laplacian{T}()
    )

Divergence{T}(flux, phi) where T = Operator(
    flux, phi, 1, Divergence{T}()
    )

# SOURCES

# Base Source
struct Src{F,S,T}
    field::F 
    sign::S 
    type::T
end

# Source types

struct Source <: AbstractSource end

Source(f::AbstractVector) = Src(f, 1, typeof(f))
Source(f::ScalarField) = Src(f.values, 1, typeof(f))
# Source(f::Number) = Src(f.values, 1, typeof(f)) # To implement!!

# MODEL TYPE
struct Model{T,S, TN, SN}
    terms::T
    sources::S
end
Model{TN,SN}(terms::T, sources::S) where {T,S,TN,SN} = begin
    Model{T,S,TN,SN}(terms, sources)
end


