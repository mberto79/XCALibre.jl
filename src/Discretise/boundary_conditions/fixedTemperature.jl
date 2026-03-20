export FixedTemperature
export Enthalpy

"""
    FixedTemperature(name::Symbol, model::Enthalpy; T::Number)

Fixed temperature boundary condition model, which allows the user to specify wall
temperature that can be translated to the energy specific model, such as sensible enthalpy.

# Inputs
- `name`: Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `T`: keyword argument use to define the boundary temperature 
- `model`: defines the underlying `Energy` model to be used (currently only `Enthalpy` is available)

# Example
    FixedTemperature(:inlet, T=300.0, Enthalpy(cp=cp, Tref=288.15))
"""
struct FixedTemperature{I,V,R<:UnitRange} <: AbstractDirichlet
    ID::I 
    value::V
    IDs_range::R 
end
Adapt.@adapt_structure FixedTemperature

@kwdef struct FixedTemperatureValue{F,M}
    T::F
    energy_model::M 
end
Adapt.@adapt_structure FixedTemperatureValue

# Temperature models (Enthalpy only for now!)
@kwdef struct Enthalpy{C,F}
    cp::C
    Tref::F
end

# API-Level constructor for FixedTemperature
FixedTemperature(name, model::Enthalpy; T) = begin
    FixedTemperature(name, FixedTemperatureValue(T=T, energy_model=model))
end

# Conversion temperature to sensible enthalpy
@inline (model::Enthalpy)(T) = begin
    cp = model.cp
    Tref = model.Tref
    h = cp*(T - Tref)
    return h
end

adapt_value(value::FixedTemperatureValue, mesh) = begin
    F = _get_float(mesh)
    FixedTemperatureValue(F(value.T), value.energy_model)
end

# DEFINITION OF BOUNDARY CONDITIONS FOR AVAILABLE SCHEMES

@define_boundary FixedTemperature Divergence{Linear} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    ap = term.sign*(-flux)
    h = energy_model(T) # To do: find nicer way to accomplish this
    0.0, ap*h
end


@define_boundary FixedTemperature Divergence{Upwind} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    h = energy_model(T)
    ap = term.sign*(flux)
    0.0, -ap*h
end

@define_boundary FixedTemperature Divergence{LUST} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    h = energy_model(T)
    ap = term.sign*(flux)
    0.0, -ap*h
end


@define_boundary FixedTemperature Divergence{BoundedUpwind} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    ap = term.sign*(flux)
    h = energy_model(T)
    -flux, -ap*h
end

@define_boundary FixedTemperature Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face 
    (; T, energy_model) = bc.value
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    h = energy_model(T)
    ap, ap*h
end

