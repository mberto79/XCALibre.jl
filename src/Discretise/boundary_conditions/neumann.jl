export Neumann


"""
    Neumann <: AbstractNeumann

Neumann boundary condition model to set the gradient at the boundary explicitly *(currently only configured for zero gradient)*

# Inputs
- `ID` Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` Scalar providing face normal gradient

# Example
    Neumann(:outlet, 0)
"""
struct Neumann{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Neumann

@define_boundary Neumann Laplacian{Linear} ScalarField begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = J*area
    0.0, flux*bc.value # draft implementation to test!
end

@define_boundary Neumann Divergence{Linear} ScalarField begin
    flux = term.flux[fID]
    (; area, delta) = face 
    ap = (flux) 
    ap, -bc.value*ap*delta
end

@define_boundary Neumann Divergence{Upwind} ScalarField begin
    flux = term.flux[fID]
    (; area, delta) = face 
    ap = (flux) 
    ap, -bc.value*ap*delta
end

@define_boundary Neumann Divergence{LUST} ScalarField begin
    flux = term.flux[fID]
    (; area, delta) = face 
    ap = (flux) 
    ap, -bc.value*ap*delta
end

@define_boundary Neumann Divergence{BoundedUpwind} ScalarField begin
    flux = term.flux[fID]
    (; area, delta) = face 
    ap = (flux) 
    ap, -bc.value*ap*delta
end

@define_boundary Neumann Si ScalarField begin
    0.0, 0.0
end