export AbstractMomentumModel, AbstractTurbulenceModel
export AbstractRANSModel
export RANS

abstract type AbstractMomentumModel end
abstract type AbstractTurbulenceModel end
abstract type AbstractRANSModel <: AbstractTurbulenceModel end

# Models 

struct RANS{T} <:AbstractRANSModel end
Adapt.Adapt.@adapt_structure RANS