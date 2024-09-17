export AbstractModelContainer
export AbstractTurbulenceModel
export AbstractRANSModel, RANS
export AbstractLESModel, LES

abstract type AbstractModelContainer end
abstract type AbstractTurbulenceModel end
abstract type AbstractRANSModel <: AbstractTurbulenceModel end
abstract type AbstractLESModel <: AbstractTurbulenceModel end

Base.show(io::IO, model::AbstractTurbulenceModel) = print(io, typeof(model).name.wrapper)

# Models 
"""
    RANS <: AbstractRANSModel

Abstract RANS model type for consturcting RANS models.

### Fields
- 'args' -- Model arguments.
"""
struct RANS{T,ARG} <:AbstractModelContainer 
    args::ARG
end

"""
    LES <: AbstractLESModel

Abstract LES model type for constructing LES models.

### Fields
- 'args' -- Model arguments.
"""
struct LES{T,ARG} <:AbstractModelContainer 
    args::ARG
end
