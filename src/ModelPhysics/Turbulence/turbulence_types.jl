export AbstractModelContainer
export AbstractTurbulenceModel
export AbstractRANSModel, RANS
export AbstractLESModel, LES
export AbstractDummyTurbulenceModel

abstract type AbstractModelContainer end
abstract type AbstractTurbulenceModel end
abstract type AbstractRANSModel <: AbstractTurbulenceModel end
abstract type AbstractLESModel <: AbstractTurbulenceModel end
abstract type AbstractDummyTurbulenceModel <: AbstractTurbulenceModel end

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

# face fields a restart rebuilds from restored cell fields, as the model's last iteration left them
restart_turbulence!(turbulence, model, config, time) = nothing

"""
    eddy_viscosity!(turbulenceModel, model, S, time, config)

Sets ν_t (cell and face) from the current fields without advancing any transport equation.
"""
eddy_viscosity!(m, model, S, time, config) = turbulence!(m, model, S, nothing, time, config)

# fields a solver treats explicitly: ν_t and, under Crank-Nicolson, the lagged copies a
# transport model takes its coefficients from
explicit_fields(m, model) = (_eddy_fields(model.turbulence)..., _lag_pairs(m, model)...)
_eddy_fields(t) = hasproperty(t, :nutf) ? (t.nut, t.nutf) : ()
_lag_pairs(m, model) = hasproperty(m, :levels) && !isnothing(m.levels) ?
    Tuple(m.levels.lag[k] => getproperty(model.turbulence, k) for k ∈ keys(m.levels.lag)) : ()

# a transport model's lagged copies and old levels of its fields, for Crank-Nicolson only
transport_levels(eqn, fields, mesh) = is_crank_nicolson_time(eqn) ?
    (lag=map(f -> ScalarField(mesh), fields), old=map(f -> ScalarField(mesh), fields)) : nothing
is_crank_nicolson_time(eqn) = unwrap_eqn(eqn).model.terms[1].type isa Time{CrankNicolson}

# coefficients from the lagged copies (kept positive), old levels kept for the Time terms
lagged_coefficients!(::Nothing, fields) = nothing
function lagged_coefficients!(levels, fields)
    foreach(levels.old, fields, levels.lag) do o, f, l
        copyto!(o.values, f.values)
        @. f.values = ifelse(l.values > 0, l.values, o.values)
    end
end
old_level(::Nothing, name, f) = f
old_level(levels, name, f) = levels.old[name]
