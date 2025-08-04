export AbstractSolid
export Solid
export Uniform

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

@kwdef struct Uniform{S1, F1, S2, S3} <: AbstractSolid
    k::S1
    kf::F1
    cp::S2
    rho::S3
end
Adapt.@adapt_structure Uniform

Solid{Uniform}(; k, cp = nothing, rho = nothing) = begin 
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

    # Build fields for solide
    k = ConstantScalar(k)
    kf = k
    cp = ConstantScalar(cp)
    rho = ConstantScalar(rho)
    Uniform(k, kf, cp, rho)
end