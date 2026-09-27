export Neumann


"""
    Neumann <: AbstractNeumann

Neumann boundary condition model to set the face normal gradient at the boundary explicitly

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
    J = term.flux[fID]
    0.0, -term.sign*J*faces.area[fID]*bc.value
end

@define_boundary Neumann Divergence{Linear} ScalarField begin
    flux = term.flux[fID]
    area, delta = faces.area[fID], faces.delta[fID]
    ap = term.sign*(flux) 
    ap, -bc.value*ap*delta
end

@define_boundary Neumann Divergence{Upwind} ScalarField begin
    flux = term.flux[fID]
    area, delta = faces.area[fID], faces.delta[fID]
    ap = term.sign*(flux) 
    ap, -bc.value*ap*delta
end

@define_boundary Neumann Divergence{LUST} ScalarField begin
    flux = term.flux[fID]
    area, delta = faces.area[fID], faces.delta[fID]
    ap = term.sign*(flux) 
    ap, -bc.value*ap*delta
end

# Bounded = upwind boundary with -Sp(div phi): subtract ap from the diagonal
@define_boundary Neumann Divergence{BoundedUpwind} ScalarField begin
    delta = faces.delta[fID]
    ap = term.sign*(term.flux[fID])
    0.0, -bc.value*ap*delta
end

@define_boundary Neumann Si ScalarField begin
    0.0, 0.0
end