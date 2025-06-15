export FixedTemperature

"""
    FixedTemperature <: AbstractDirichlet

Fixed temperature boundary condition model, which allows the user to specify wall
temperature that can be translated to the energy specific model, such as sensivle enthalpy.

### Fields
- 'ID' -- Boundary ID
- `value` -- Scalar or Vector value for Dirichlet boundary condition.
- `T` - Temperature value in Kelvin.
- `model` - Energy physics model for case.

### Examples
    FixedTemperature(:inlet, T=300.0, model=model.energy),
"""
struct FixedTemperature{S,V,I} <: AbstractDirichlet
    name::S
    value::V
    ID::I 
    IDs_range::UnitRange{I} 
end
Adapt.@adapt_structure FixedTemperature

FixedTemperature(name; T, model) = begin
    FixedTemperature(name, (; T=T, energy_model=model))
end

function fixedValue(BC::FixedTemperature, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return FixedTemperature{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return FixedTemperature{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
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

