export Energy
export AbstractEnergyModel
export Isothermal

abstract type AbstractEnergyModel end

# Models

struct Energy{T,ARG}
    args::ARG
end

# Isothermal
struct Isothermal <: AbstractEnergyModel end

Energy{Isothermal}() = begin
    args = nothing
    ARGS = typeof(args)
    Energy{Isothermal,ARGS}(args)
end