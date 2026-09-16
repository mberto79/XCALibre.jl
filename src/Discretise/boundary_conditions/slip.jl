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

# Face value = tangential projection vp = vc - (vc⋅n)n. Split the same-component
# term implicitly on outflow, defer cross-components and inflow to the source.
@define_boundary Slip Divergence{Upwind} VectorField begin
    (; normal) = face
    phi = term.phi
    ap = term.sign*(term.flux[fID])

    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    vp = vc - vn

    nc = normal[component.value]
    vc_c = vc[component.value]
    vp_c = vp[component.value]
    z = zero(ap)
    one_minus_nc2 = one(nc) - nc^2

    ac = max(ap, z) * one_minus_nc2
    su_leaving = -max(ap, z) * (vp_c - vc_c * one_minus_nc2)
    su_entering = -min(ap, z) * vp_c
    ac, su_entering + su_leaving
end

@define_boundary Slip Divergence{Upwind} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    z = zero(ap)
    ac = max(ap, z)
    su = -min(ap, z) * get_values(term.phi, component)[cellID]
    ac, su
end

# Impermeable wall: face value = cell value, so div (+ap) and bounding (-ap) cancel
@define_boundary Slip Divergence{BoundedUpwind} VectorField begin
    0.0, 0.0
end

@define_boundary Slip Divergence{BoundedUpwind} ScalarField begin
    0.0, 0.0
end


@define_boundary Slip Divergence{Linear} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    z = zero(ap)
    ac = max(ap, z)
    su = -min(ap, z) * get_values(term.phi, component)[cellID]
    ac, su
end

@define_boundary Slip Divergence{LUST} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    z = zero(ap)
    ac = max(ap, z)
    su = -min(ap, z) * get_values(term.phi, component)[cellID]
    ac, su
end

@define_boundary Slip Divergence{Linear} VectorField begin
    (; normal) = face 
    phi = term.phi
    flux = term.flux[fID]
    ap = term.sign*(flux)       # = ϕ_b
    
    # Tangential projection
    vc = phi[cellID]
    vn = (vc ⋅ normal) * normal
    vp = vc - vn
    
    nc = normal[component.value]
    vc_c = vc[component.value]
    vp_c = vp[component.value]
    z = zero(ap)
    one_minus_nc2 = one(nc) - nc^2
    
    # Outflow (ap > 0): implicit same-component, defer cross terms
    ac = max(ap, z) * one_minus_nc2
    su_leaving = -max(ap, z) * (vp_c - vc_c * one_minus_nc2)
    
    # Inflow (ap < 0): defer everything to source
    su_entering = -min(ap, z) * vp_c
    
    ac, su_entering + su_leaving
end

@define_boundary Slip Divergence{LUST} VectorField begin
    (; normal) = face 
    phi = term.phi
    flux = term.flux[fID]
    ap = term.sign*(flux)
    
    vc = phi[cellID]
    vn = (vc ⋅ normal) * normal
    vp = vc - vn
    
    nc = normal[component.value]
    vc_c = vc[component.value]
    vp_c = vp[component.value]
    z = zero(ap)
    one_minus_nc2 = one(nc) - nc^2
    
    ac = max(ap, z) * one_minus_nc2
    su_leaving = -max(ap, z) * (vp_c - vc_c * one_minus_nc2)
    su_entering = -min(ap, z) * vp_c
    
    ac, su_entering + su_leaving
end
