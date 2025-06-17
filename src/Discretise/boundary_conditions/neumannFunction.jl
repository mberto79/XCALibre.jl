export NeumannFunction

# abstract type XCALibreUserFunctor end

"""
    NeumannFunction(ID, value) <: AbstractNeumann

Neumann boundary condition defined with user-provided function.

# Input
- `ID` Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` Custom function defining the desired Neumann boundary condition.

# Function requirements

The function passed to this boundary condition has not yet been implemented. However, users can pass a custom struct to specialise the internal implementations of many functions. By default, at present, this function will assign a zero gradient boundary condition.
"""
struct NeumannFunction{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure NeumannFunction

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