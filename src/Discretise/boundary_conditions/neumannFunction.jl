export NeumannFunction

# abstract type XCALibreUserFunctor end

"""
    NeumannFunction(ID, value) <: AbstractNeumann

Neumann boundary condition defined with user-provided function.

# Input
- `ID` Boundary name provided as symbol e.g. :inlet
- `value` Custom function for Neumann boundary condition.

# Function requirements

The function passed to this boundary condition has not yet been implemented. However, users can pass a custom struct to specialise the internal implementations of many functions. By default, as present, this function will assign a zero gradient boundary condition on all fields.
"""
struct NeumannFunction{S,V,I} <: AbstractNeumann
    name::S
    value::V
    ID::I 
    IDs_range::UnitRange{I}
end
Adapt.@adapt_structure NeumannFunction


function fixedValue(BC::NeumannFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: value is scalar
    if V <: Number
        return NeumannFunction{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return NeumannFunction{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid        
        else
            throw("Only vectors with three components can be used")
        end
    # Exception 3: value is a function
    elseif V <: Function
        return NeumannFunction{I,V}(ID, value)
    # Exception 4: value is a user provided XCALibre functor
    elseif V <: XCALibreUserFunctor
        return NeumannFunction{I,V}(ID, value)
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

@define_boundary NeumannFunction Laplacian{Linear} begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
    ap, ap*values[cellID] # original
    0.0, 0.0 
    # 0.0, -flux*delta*bc.value # draft implementation to test!
end

@define_boundary NeumannFunction Divergence{Linear} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary NeumannFunction Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary NeumannFunction Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary NeumannFunction Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

@define_boundary NeumannFunction Si begin
    0.0, 0.0
end