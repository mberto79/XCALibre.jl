export Neumann


"""
    Neumann <: AbstractNeumann

Neumann boundary condition model *(currently only configured for zero gradient)*

### Fields
- 'ID' -- Boundary ID
- `value` -- Scalar or Vector value for Neumann boundary condition.
"""
struct Neumann{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Neumann

@define_boundary Neumann Laplacian{Linear} begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
    ap, ap*values[cellID] # original
    # 0.0, 0.0 # try this
    # 0.0, -flux*bc.value # draft implementation to test!
end

@define_boundary Neumann Divergence{Linear} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Neumann Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Neumann Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Neumann Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

@define_boundary Neumann Si begin
    0.0, 0.0
end