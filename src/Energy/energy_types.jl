export AbstractEnergyModel, ENERGY
export Isothermal

abstract type AbstractEnergyModel end

# Models

struct ENERGY{T,ARG} <: AbstractEnergyModel
    args::ARG
end

# Isothermal

struct Isothermal end
Adapt.Adapt.@adapt_structure Isothermal

ENERGY{Isothermal}() = begin
    args = nothing
    ARGS = typeof(args)
    ENERGY{Isothermal,ARGS}(args)
end

(energy::ENERGY{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:Isothermal,ARG} = begin
    nothing
end