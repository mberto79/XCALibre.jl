export AbstractTurbulenceModel
export AbstractRANSModel, RANS
export AbstractLESModel, LES

abstract type AbstractTurbulenceModel end
abstract type AbstractRANSModel <: AbstractTurbulenceModel end
abstract type AbstractLESModel <: AbstractTurbulenceModel end

# Models 
"""
    RANS <: AbstractRANSModel

Abstract RANS model type for consturcting RANS models.

### Fields
- 'args' -- Model arguments.
"""
struct RANS{T,ARG} <:AbstractRANSModel 
    args::ARG
end

"""
    LES <: AbstractLESModel

Abstract LES model type for constructing LES models.

### Fields
- 'args' -- Model arguments.
"""
struct LES{T,ARG} <:AbstractLESModel 
    args::ARG
end
