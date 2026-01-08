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

struct Isothermal{T<:AbstractScalarField} <: AbstractEnergyModel 
    T::T
end 
Adapt.Adapt.@adapt_structure Isothermal

# Set the defualt value unless user specifies it - compatible with other solvers and energy models.
Energy{Isothermal}(; T=300.0) = begin
    coeffs = (T=T,)
    ARG = typeof(coeffs)
    Energy{Isothermal,ARG}(coeffs)
end

(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:Isothermal,ARG} = begin
    T_val = energy.args.T
    
    T = ConstantScalar(T_val)
    Isothermal(T)
end