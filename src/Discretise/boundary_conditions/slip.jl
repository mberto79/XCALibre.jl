export Slip

"""
    Slip <: AbstractPhysicalConstraint

Slip boundary condition for vector and scalar fields. Vectors keep both
tangential components and have the face-normal component removed, so the patch
is impermeable but exerts no tangential shear. Scalars use an explicit
zero-normal-gradient condition. Use `Wall` instead when the patch should apply
no-slip.

# Input
- `ID` is the boundary name (for example, `:plate`). It is replaced by the
  boundary index during boundary assignment.

# Example
    Slip(:plate)
"""
struct Slip{I,V,R<:UnitRange} <: AbstractPhysicalConstraint
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Slip

Slip(name::Symbol) = Slip(name, 0)

@define_boundary Slip Laplacian{Linear} ScalarField begin
    0.0, 0.0
end

@define_boundary Slip Laplacian{Linear} VectorField begin
    (; area, delta, normal) = face
    J = term.flux[fID]
    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    vc = term.phi[cellID]
    vp = vc - (vc⋅normal)*normal
    # ac = ap (not ap*nc^2) buys diagonal dominance; the deferred source cancels exactly at convergence
    ap, ap*vp[component.value]
end

@define_boundary Slip Divergence{Upwind} VectorField begin
    ap = term.sign*term.flux[fID]
    _tangential_divergence(ap, term.phi[cellID], face.normal, component)
end

@define_boundary Slip Divergence{Upwind} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    z = zero(ap)
    ac = max(ap, z)
    su = -min(ap, z) * get_values(term.phi, component)[cellID]
    ac, su
end

# Scalars cancel exactly. For vectors, the projected face value leaves the normal
# component from div(phi,U) - Sp(div(phi),U).
@define_boundary Slip Divergence{BoundedUpwind} VectorField begin
    (; normal) = face
    ap = term.sign*term.flux[fID]
    vc = term.phi[cellID]
    vn = (vc⋅normal)*normal
    0.0, ap*vn[component.value]
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
    ap = term.sign*term.flux[fID]
    _tangential_divergence(ap, term.phi[cellID], face.normal, component)
end

@define_boundary Slip Divergence{LUST} VectorField begin
    ap = term.sign*term.flux[fID]
    _tangential_divergence(ap, term.phi[cellID], face.normal, component)
end

@define_boundary Slip Si begin
    0.0, 0.0
end
