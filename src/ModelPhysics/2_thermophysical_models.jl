export HelmholtzEnergy, HelmholtzEnergyFluid, H2, H2_para, N2
export ConstEos, ConstMu
export AbstractModel, AbstractEosModel, AbstractViscosityModel


abstract type AbstractModel end
abstract type AbstractEosModel <: AbstractModel end
abstract type AbstractViscosityModel <: AbstractModel end

abstract type AbstractPhase <: AbstractMultiphase end


abstract type HelmholtzEnergyFluid end

struct N2 <: HelmholtzEnergyFluid end
struct H2 <: HelmholtzEnergyFluid end
struct H2_para <: HelmholtzEnergyFluid end

Base.@kwdef struct HelmholtzEnergy{F<:HelmholtzEnergyFluid}
    name::F
end

Base.@kwdef struct HydrogenViscosity <: AbstractPhysicsProperty end
Base.@kwdef struct NitrogenViscosity <: AbstractPhysicsProperty end


"""
    ConstEos <: AbstractEosModel

Constant density equation of state model.

### Fields
- 'rho' -- Constant density value.
"""
Base.@kwdef struct ConstEos{T<:AbstractFloat} <: AbstractEosModel
    rho::T
end
(eos::ConstEos)(phase, model, config) = begin
    rho_field = phase.rho
    initialise!(rho_field, eos.rho)
end


"""
    ConstMu <: AbstractViscosityModel

Constant dynamic viscosity model.

### Fields
- 'mu' -- Dynamic viscosity value [Pa⋅s].
"""
Base.@kwdef struct ConstMu{T<:AbstractFloat} <: AbstractViscosityModel
    mu::T
end
(mu::ConstMu)(phase, model) = begin
    nu_field = phase.nu
    mu_val = phase.mu.mu
    rho_val = phase.density.rho
    
    initialise!(nu_field, mu_val/rho_val)
end