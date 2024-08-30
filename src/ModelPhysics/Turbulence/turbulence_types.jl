export AbstractMomentumModel, AbstractTurbulenceModel
export AbstractRANSModel, RANS
export AbstractLESModel, LES

abstract type AbstractMomentumModel end
abstract type AbstractTurbulenceModel end
abstract type AbstractRANSModel <: AbstractTurbulenceModel end
abstract type AbstractLESModel <: AbstractTurbulenceModel end

# Models 

struct RANS{T,ARG} <:AbstractRANSModel 
    args::ARG
end

struct LES{T,ARG} <:AbstractLESModel 
    args::ARG
end
