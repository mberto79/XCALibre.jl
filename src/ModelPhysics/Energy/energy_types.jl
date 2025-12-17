# export AbstractEnergyModel, Energy
export Energy
export AbstractEnergyModel
export Isothermal

abstract type AbstractEnergyModel end

# Models

struct Energy{T,ARG}
    args::ARG
end

# Isothermal

#### THS IS A DUMMY IMOLEENTATIUO PLEASE SORT OUT ARTEM! TA
struct Isothermal{T<:AbstractFloat} <: AbstractEnergyModel 
    T_iso::T
end 
Adapt.Adapt.@adapt_structure Isothermal


# These constructors might need changing
Energy{Isothermal}() = begin
    args = nothing
    ARGS = typeof(args)
    Energy{Isothermal,ARGS}(args)
end

(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:Isothermal,ARG} = begin
    nothing
end