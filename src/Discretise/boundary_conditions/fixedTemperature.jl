export FixedTemperature

"""
    struct FixedTemperature{I,V,R<:UnitRange} <: AbstractDirichlet
        ID::I 
        value::V
        IDs_range::R 
    end

Fixed temperature boundary condition model, which allows the user to specify wall
temperature that can be translated to the energy specific model, such as sensible enthalpy.

# Inputs
- `ID` Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `T` is used to set the boundary temperature
- `model` defines the underlying `Energy` model to be used

# Example
    FixedTemperature(:inlet, T=300.0, model=model.energy)
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

adapt_value(value::FixedTemperatureValue, mesh) = begin
    F = _get_float(mesh)
    FixedTemperatureValue(F(value.T), value.energy_model)
end

FixedTemperature(name; T, model) = begin
    FixedTemperature(name, FixedTemperatureValue(T=T, energy_model=model))
end

@define_boundary FixedTemperature Divergence{Linear} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    ap = term.sign*(-flux)
    h = energy_model.update_BC(T) # To do: find nicer way to accomplish this
    0.0, ap*h
end


@define_boundary FixedTemperature Divergence{Upwind} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    h = energy_model.update_BC(T)
    ap = term.sign*(flux)
    0.0, -ap*h
end

@define_boundary FixedTemperature Divergence{LUST} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    h = energy_model.update_BC(T)
    ap = term.sign*(flux)
    0.0, -ap*h
end


@define_boundary FixedTemperature Divergence{BoundedUpwind} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    ap = term.sign*(flux)
    h = energy_model.update_BC(T)
    -flux, -ap*h
end

@define_boundary FixedTemperature Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face 
    (; T, energy_model) = bc.value
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    h = energy_model.update_BC(T)
    ap, ap*h
end

