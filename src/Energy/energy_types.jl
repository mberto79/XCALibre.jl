export AbstractEnergyModel, ENERGY
export Isothermal

abstract type AbstractEnergyModel end

# Models

struct ENERGY{T,ARG} <: AbstractEnergyModel
    args::ARG
end

# Isothermal

struct Isothermal end

ENERGY{Isothermal}() = ENERGY{Isothermal}(nothing)

(energy::ENERGY{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:Isothermal,ARG} = begin
    nothing
end