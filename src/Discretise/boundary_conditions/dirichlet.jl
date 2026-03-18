export Dirichlet

"""
    Dirichlet <: AbstractDirichlet

Dirichlet boundary condition model.

# Inputs
- `ID` Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` Scalar or Vector value for Dirichlet boundary condition
"""
struct Dirichlet{I,V,R<:UnitRange} <: AbstractDirichlet
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Dirichlet

@define_boundary Dirichlet Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face 
    flux = J*area/delta
    ap = term.sign*(-flux)
    ap, ap*bc.value
end

@define_boundary Dirichlet Divergence{Linear} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{Upwind} begin
    phi = term.sign * term.flux[fID]   # signed flux, consistent with BoundedUpwind
    ap = max(phi, 0.0)                 # outflow → diagonal (positive)
    su = max(-phi, 0.0) * bc.value     # inflow → source (positive)
    ap, su
end

@define_boundary Dirichlet Divergence{LUST} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Laplacian{Linear} VectorField begin
    J = term.flux[fID]
    (; area, delta) = face 
    flux = J*area/delta
    ap = term.sign*(-flux)
    ap, ap*bc.value[component.value]
end

@define_boundary Dirichlet Divergence{Linear} VectorField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value[component.value]
end

@define_boundary Dirichlet Divergence{Upwind} VectorField begin
    phi = term.sign * term.flux[fID]
    ap = max(phi, 0.0)
    su = max(-phi, 0.0) * bc.value[component.value]
    ap, su
end

@define_boundary Dirichlet Divergence{LUST} VectorField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value[component.value]
end

@define_boundary Dirichlet Divergence{BoundedUpwind} VectorField begin
    # ap = term.sign*(term.flux[fID])
    # ac = max(-ap, 0.0)
    # phic = get_values(term.phi, component)[cellID]
    # 0.0, -ap*bc.value[component.value]

    ap = term.sign*(term.flux[fID])
    ac = max(-ap, 0.0)
    an = -max(-ap, 0.0)
    ac, -an*bc.value[component.value]
end

@define_boundary Dirichlet Divergence{BoundedUpwind} begin
    # ap = term.sign*(term.flux[fID])
    # ac = max(-ap, 0.0)
    # phic = get_values(term.phi, component)[cellID]
    # 0.0, -ap*bc.value

    ap = term.sign*(term.flux[fID])
    ac = max(-ap, 0.0)
    an = -max(-ap, 0.0)
    ac, -an*bc.value
end

@define_boundary Dirichlet Si begin
    0.0, 0.0
end
