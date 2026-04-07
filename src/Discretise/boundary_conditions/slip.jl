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
    # (; normal) = face 
    # phi = term.phi
    # flux = term.flux[fID]
    # ap = term.sign*(flux) 
    # vc = phi[cellID]
    # vn = (vc⋅normal)*normal
    # vp = vc - vn
    # 0.0, -ap*vp[component.value]

    (; normal) = face 
    phi = term.phi
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    
    # Extract cell vector and compute strictly tangential slip vector
    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    vp = vc - vn
    
    # Component-specific scalars
    nc = normal[component.value]
    vc_c = vc[component.value]
    vp_c = vp[component.value]
    
    # 1. Flow leaving (ap > 0): Implicit tensorial split
    # Diagonal is scaled by (1 - n^2) to strictly strip the normal contribution.
    ac = max(ap, 0.0) * (1.0 - nc^2)
    
    # The remainder of the slip vector (cross-components) goes to deferred correction
    su_leaving = -max(ap, 0.0) * (vp_c - vc_c * (1.0 - nc^2))
    
    # 2. Flow entering (ap < 0): Explicit Upwind Boundary evaluation
    su_entering = -min(ap, 0.0) * vp_c
    
    # Total explicit source
    su = su_entering + su_leaving
    
    ac, su
    0.0, -ap*get_values(term.phi, component)[cellID]
end

# @define_boundary Slip Divergence{Upwind} ScalarField begin
#     # flux = term.flux[fID]
#     # ap = term.sign*(flux) 
#     # ap, 0.0 # original

#     flux = term.flux[fID]
#     ap = max(term.sign*(flux), 0.0)
#     ap, 0.0 # original
# end

@define_boundary Slip Divergence{Upwind} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ac = max(ap, 0.0)
    su = -min(ap, 0.0) * get_values(term.phi, component)[cellID]
    ac, su
end

# @define_boundary Slip Divergence{BoundedUpwind} VectorField begin
#     # (; normal) = face 
#     # phi = term.phi
#     # flux = term.flux[fID]
#     # ap = term.sign*(flux) 
#     # vc = phi[cellID]
#     # vn = (vc⋅normal)*normal
#     # vp = vc - vn
#     # 0.0, -ap*vp[component.value]
#     (; normal) = face 
#     phi = term.phi
#     flux = term.flux[fID]
#     ap = term.sign*(flux) 
    
#     vc = phi[cellID]
#     vn = (vc⋅normal)*normal
#     vp = vc - vn
    
#     nc = normal[component.value]
#     vc_c = vc[component.value]
#     vp_c = vp[component.value]

#     # BoundedUpwind logic mirrors the diagonal logic but with opposite sign 
#     # for the implicit part to match the internal term: Σ φf ψf - ψp Σ φf
    
#     # 1. Flow entering (ap < 0): Becomes implicit on the diagonal
#     ac = max(-ap, 0.0) * (1.0 - nc^2)
    
#     # 2. Source from entering flow
#     su_entering = max(-ap, 0.0) * vp_c
    
#     # 3. Source correction to remove cross-components from the diagonal
#     su_correction = -max(-ap, 0.0) * (vp_c - vc_c * (1.0 - nc^2))
    
#     ac, su_entering + su_correction
# end

@define_boundary Slip Divergence{BoundedUpwind} VectorField begin
    (; normal) = face 
    phi = term.phi
    flux = term.flux[fID]
    ap = term.sign*(flux)       # = ϕ_b
    
    vc = phi[cellID]
    nc = normal[component.value]
    
    # Bounded form: ϕ_b(v_{f,i} - v_{P,i}) = -ϕ_b·n_i²·v_{P,i} + cross terms
    # Only inflow contributes safely to diagonal
    
    F_upwind = max(-ap, 0.0)           # |ϕ_b| on inflow, 0 on outflow
    
    # Implicit: diagonal coefficient from same-component
    ac = F_upwind * nc^2
    
    # Cross-component source: -ϕ_b·n_i·Σ_{j≠i} n_j·v_{P,j}
    # = F·n_i·Σ_{j≠i} n_j·v_{P,j}  (on inflow)
    vn_cross = (vc ⋅ normal) - vc[component.value] * nc  # = Σ_{j≠i} n_j·v_{P,j}
    su_cross = F_upwind * nc * vn_cross
    
    ac, su_cross
end

@define_boundary Slip Divergence{BoundedUpwind} ScalarField begin
    # ap = term.sign*(term.flux[fID])
    # ac = max(-ap, 0.0)
    # an = -max(-ap, 0.0)
    # ac, -an*bc.value
    0.0, 0.0
end


@define_boundary Slip Divergence{Linear} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ac = max(ap, 0.0)
    su = -min(ap, 0.0) * get_values(term.phi, component)[cellID]
    ac, su
end

@define_boundary Slip Divergence{LUST} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ac = max(ap, 0.0)
    su = -min(ap, 0.0) * get_values(term.phi, component)[cellID]
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
    
    # Outflow (ap > 0): implicit same-component, defer cross terms
    ac = max(ap, 0.0) * (1.0 - nc^2)
    su_leaving = -max(ap, 0.0) * (vp_c - vc_c * (1.0 - nc^2))
    
    # Inflow (ap < 0): defer everything to source
    su_entering = -min(ap, 0.0) * vp_c
    
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
    
    ac = max(ap, 0.0) * (1.0 - nc^2)
    su_leaving = -max(ap, 0.0) * (vp_c - vc_c * (1.0 - nc^2))
    su_entering = -min(ap, 0.0) * vp_c
    
    ac, su_entering + su_leaving
end