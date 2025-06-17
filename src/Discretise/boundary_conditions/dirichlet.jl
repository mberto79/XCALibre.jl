export Dirichlet

"""
    Dirichlet <: AbstractDirichlet

Dirichlet boundary condition model.

# Fields
- 'ID' -- Boundary ID
- `value` -- Scalar or Vector value for Dirichlet boundary condition.
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
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{LUST} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{BoundedUpwind} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    flux, ap*bc.value
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
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value[component.value]
end

@define_boundary Dirichlet Divergence{LUST} VectorField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value[component.value]
end

@define_boundary Dirichlet Divergence{BoundedUpwind} VectorField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    flux, ap*bc.value[component.value]
end

@define_boundary Dirichlet Si begin
    0.0, 0.0
end
