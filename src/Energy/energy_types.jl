export AbstractEnergyModel, ENERGY

abstract type AbstractEnergyModel end

# Models

struct ENERGY{T,ARG} <: AbstractEnergyModel
    args::ARG
end