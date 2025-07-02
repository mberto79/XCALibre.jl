export AbstractModelContainer
export AbstractTurbulenceModel
export AbstractRANSModel, RANS
export AbstractLESModel, LES

# export NoTurbulence

abstract type AbstractModelContainer end
abstract type AbstractTurbulenceModel end
abstract type AbstractRANSModel <: AbstractTurbulenceModel end
abstract type AbstractLESModel <: AbstractTurbulenceModel end

# abstract type NoTurb <: AbstractTurbulenceModel end

Base.show(io::IO, model::AbstractTurbulenceModel) = print(io, typeof(model).name.wrapper)

# Models 


# struct NoTurbulence{T,ARG} <:AbstractModelContainer 
#     args::ARG
# end



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
