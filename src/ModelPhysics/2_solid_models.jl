export AbstractSolid
export Solid
export Uniform, NonUniform

abstract type AbstractSolid end

Base.show(io::IO, solid::AbstractSolid) = print(io, typeof(solid).name.wrapper)


"""
    Solid <: AbstractSolid

Abstract solid model type for constructing new solid models.

### Fields
- 'args' -- Model arguments.

"""
struct Solid{T,ARG}
    args::ARG
end

@kwdef struct Uniform{S1, F1, S2, S3} <: AbstractSolid #k, cp, rho are the same everywhere in the domain
    k::S1
    kf::F1
    cp::S2
    rho::S3
end
Adapt.@adapt_structure Uniform

Solid{Uniform}(; k, cp=nothing, rho=nothing) = begin 
    coeffs = (k=k, cp, rho,)
    ARG = typeof(coeffs)
    Solid{Uniform,ARG}(coeffs)
end

(solid::Solid{Uniform, ARG})(mesh, time) where ARG = begin
    coeffs = solid.args
    (; k, cp, rho) = coeffs
    
    if typeof(time) == Transient
        @assert cp !== nothing "For transient simulations cp must be provided"
        @assert rho !== nothing "For transient simulations rho must be provided"
    end

    # Build fields for solid
    k = ConstantScalar(k)
    kf = k
    cp = ConstantScalar(cp)
    rho = ConstantScalar(rho)
    Uniform(k, kf, cp, rho)
end





@kwdef struct NonUniform{S1, F1, S2, S3} <: AbstractSolid #k, cp, rho vary across the domain
    k::S1
    kf::F1
    cp::S2
    rho::S3
end
Adapt.@adapt_structure NonUniform

Solid{NonUniform}(; k, cp=nothing, rho=nothing) = begin 
    coeffs = (k=k, cp, rho,)
    ARG = typeof(coeffs)
    Solid{NonUniform,ARG}(coeffs)
end

(solid::Solid{NonUniform, ARG})(mesh, time) where ARG = begin
    coeffs = solid.args
    (; k, cp, rho) = coeffs
    
    if typeof(time) == Transient
        @assert cp !== nothing "For transient simulations cp must be provided"
        @assert rho !== nothing "For transient simulations rho must be provided"
    end

    # Build fields for solid
    k = ConstantScalar(k)
    kf = k
    cp = ConstantScalar(cp)
    rho = ConstantScalar(rho)
    NonUniform(k, kf, cp, rho)
end