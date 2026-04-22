export HelmholtzEnergy, HelmholtzEnergyFluid, H2, H2_para, N2
export ConstEos, ConstMu


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
    mu_field = phase.mu
    mu_val = phase.mu[1]
    rho_val = phase.rho[1]
    
    initialise!(mu_field, mu_val/rho_val)
end