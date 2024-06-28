export Laminar

# Model type definition (hold fields)
struct Laminar <: AbstractTurbulenceModel end 
Adapt.@adapt_structure Laminar

# Model type definition (hold equation definitions and internal data)
struct LaminarModel <: AbstractTurbulenceModel end 
Adapt.@adapt_structure LaminarModel

# Model API constructor (pass user input as keyword arguments and process if needed)
RANS{Laminar}() = begin # Empty constructor
    args = (); ARG = typeof(args)
    RANS{Laminar,ARG}(args)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::RANS{Laminar, ARG})(mesh) where ARG = Laminar()

# Model initialisation
function initialise(
    turbulence::Laminar, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}
    return LaminarModel()
end

# Model solver call (implementation)
function turbulence!(rans::LaminarModel, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, config
    ) where {T,F,M,Tu<:Laminar,E,D,BI}
    nothing
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, name) where {T,F,M,Tu<:Laminar,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
        ("T", model.energy.T)
    )
    write_vtk(name, model.domain, args...)
end