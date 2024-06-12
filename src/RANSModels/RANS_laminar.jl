export Laminar

# Model type definition
struct Laminar <: AbstractTurbulenceModel end 
Adapt.@adapt_structure Laminar

# Model API constructor
RANS{Laminar}(mesh) = Laminar()

# Model initialisation
function initialise(
    turbulence::Laminar, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}
    return model
end

# Model solver call
function turbulence!(model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, config
    ) where {T,F,M,Tu<:Laminar,E,D,BI}
    nothing
end