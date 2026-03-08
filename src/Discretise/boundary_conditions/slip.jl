export Slip

"""
    Slip <: AbstractDirichlet

Slip boundary condition model for no-slip  or moving walls (linear motion). It should be applied to the velocity vector, and in most cases, its scalar variant should be applied to scalars.

# Inputs
- `ID` represents the name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` should be given as a vector for the velocity e.g. [10,0,0]. For scalar fields such as the pressure the value entry can be omitted or set to zero explicitly.

# Examples
    Slip(:plate) # slip wall condition
"""
struct Slip{I,V,R<:UnitRange} <: AbstractPhysicalConstraint
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Slip

Slip(name::Symbol) = Slip(name, 0)

@define_boundary Slip Laplacian{Linear} begin
    0.0, 0.0
end

@define_boundary Slip Divergence{Upwind} VectorField begin
    (; normal) = face 
    phi = term.phi
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    vp = vc - vn
    0.0, -ap*vp[component.value]
end

@define_boundary Slip Divergence{Upwind} ScalarField begin
    phi = term.phi 
    values = get_values(phi, component)
    flux = -term.flux[fID]
    ap = term.sign*(flux) 
    max(ap,0.0), min(ap, 0.0)*values[cellID]
end