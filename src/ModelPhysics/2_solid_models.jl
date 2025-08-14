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

@kwdef struct Uniform{S1, F1, S2, S3, S4, F2} <: AbstractSolid
    k::S1
    kf::F1
    cp::S2
    rho::S3
    rhocp::S4
    rDf::F2
end
Adapt.@adapt_structure Uniform

Solid{Uniform}(; k, cp=nothing, rho=nothing) = begin 
    coeffs = (; k=k, cp, rho)
    ARG = typeof(coeffs)
    Solid{Uniform,ARG}(coeffs)
end

(solid::Solid{Uniform, ARG})(mesh, time) where ARG = begin
    coeffs = solid.args
    (; k, cp, rho) = coeffs
    
    if typeof(time) == Transient
        @assert cp !== nothing "For transient simulations cp must be provided"
        @assert rho !== nothing "For transient simulations rho must be provided"
    else
        cp = 0.0 # in case user passed it
        rho = 0.0 # in case user passed it
    end

    k_const = k
    cp_const = cp
    rho_const = rho
    
    
    
    k = ConstantScalar(k_const)
    kf = FaceScalarField(mesh) # QUESTION: Considering this is a face... should I do this?
    initialise!(kf, k_const)

    cp = ConstantScalar(cp)
    rho = ConstantScalar(rho)
    
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0/k_const)

    rhocp_const = rho_const * cp_const
    rhocp  = ConstantScalar(rhocp_const)

    Uniform(k, kf, cp, rho, rhocp, rDf)
end



@kwdef struct NonUniform{S1, F1, S2, S3, M<:AbstractMaterial, S4, F2} <: AbstractSolid
    k::S1
    kf::F1
    cp::S2
    rho::S3
    material::M
    rhocp::S4
    rDf::F2
end
Adapt.@adapt_structure NonUniform

Solid{NonUniform}(; material, rho) = begin 
    coeffs = (; material, rho)
    ARG = typeof(coeffs)
    Solid{NonUniform,ARG}(coeffs)
end

(solid::Solid{NonUniform, ARG})(mesh, time) where ARG = begin
    coeffs = solid.args
    (; material, rho) = coeffs

    @assert typeof(time) !== Steady "NonUniform Solid requires transient time term"

    rho_const = rho


    k = ScalarField(mesh)
    kf = FaceScalarField(mesh)

    cp = ScalarField(mesh)
    rho = ScalarField(mesh)
    initialise!(rho, rho_const)
    
    rDf = FaceScalarField(mesh)
    rhocp  = ScalarField(mesh)

    NonUniform(k, kf, cp, rho, material, rhocp, rDf)
end