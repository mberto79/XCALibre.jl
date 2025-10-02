# export AbstractEnergyModel, Energy
export Energy
export Isothermal

# abstract type AbstractEnergyModel end

# Models

struct Energy{T,ARG}
    args::ARG
end

# Isothermal

struct Isothermal <: AbstractEnergyModel end 
Adapt.Adapt.@adapt_structure Isothermal

Energy{Isothermal}() = begin
    args = nothing
    ARGS = typeof(args)
    Energy{Isothermal,ARGS}(args)
end

(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:Isothermal,ARG} = begin
    nothing
end