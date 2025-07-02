export NoTurbulence

struct NoTurbulence <: AbstractRANSModel end
Adapt.@adapt_structure NoTurbulence


struct NoTurbulenceModel{S}
    state::S # required field for all turbulence models
end
Adapt.@adapt_structure NoTurbulenceModel


RANS{NoTurbulence}() = begin
    args = () ; ARG = typeof(args)
    RANS{NoTurbulence,ARG}(args)
end

(rans::RANS{NoTurbulence,ARG})(mesh) where ARG = NoTurbulence()

function initialise(
        ::NoTurbulence,
        model::Physics{T,ME,M,Tu,E,D,BI},
        mdotf, peqn, config
    ) where {T,ME,M,Tu,E,D,BI}
    state = ModelState((), true)
    return NoTurbulenceModel(state)
end

function turbulence!(
        ::NoTurbulenceModel,
        model::Physics{T,ME,M,Tu,E,D,BI},
        S, prev, time, config
    ) where {T,ME,M,Tu<:AbstractTurbulenceModel,E,D,BI}
    # Intentionally left blank – nothing to update.
    nothing
end


function save_output(model::Physics{T,ME,M,Tu,E,D,BI}, outputWriter, iteration, config
    ) where {T,ME,M,Tu<:NoTurbulence,E,D,BI}
    
    args = (
        ("T", model.energy.T),
    )
    write_results(iteration, model.domain, outputWriter, config.boundaries, args...)
end