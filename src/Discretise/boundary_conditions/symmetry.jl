export Symmetry

"""
    Symmetry <: AbstractPhysicalConstraint

Symmetry boundary condition for vector and scalar fields. Scalars use an explicit
zero-normal-gradient condition. Vectors remove the face-normal component while
retaining both tangential components.

# Input
- `ID` is the boundary name (for example, `:freestream`). It is replaced by the
  boundary index during boundary assignment.

# Example
    Symmetry(:freestream)
"""
struct Symmetry{I,V,R<:UnitRange} <: AbstractPhysicalConstraint
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Symmetry

Symmetry(patch::Symbol) = Symmetry(patch, 0)

@define_boundary Symmetry Laplacian{Linear} VectorField begin
    (; area, delta, normal) = face
    J = term.flux[fID]
    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    vc = term.phi[cellID]
    vp = vc - (vc⋅normal)*normal
    # ac = ap (not ap*nc^2) buys diagonal dominance; the deferred source cancels exactly at convergence
    ap, ap*vp[component.value]
end

@define_boundary Symmetry Laplacian{Linear} ScalarField begin
    0.0, 0.0
end

# A scalar symmetry face has the owner-cell value. Treat an unexpected incoming
# face flux explicitly; the normal flux is subsequently projected to zero.
@define_boundary Symmetry Divergence{Linear} ScalarField begin
    ap = term.sign*term.flux[fID]
    z = zero(ap)
    max(ap, z), -min(ap, z)*get_values(term.phi, component)[cellID]
end

@define_boundary Symmetry Divergence{Upwind} ScalarField begin
    ap = term.sign*term.flux[fID]
    z = zero(ap)
    max(ap, z), -min(ap, z)*get_values(term.phi, component)[cellID]
end

@define_boundary Symmetry Divergence{LUST} ScalarField begin
    ap = term.sign*term.flux[fID]
    z = zero(ap)
    max(ap, z), -min(ap, z)*get_values(term.phi, component)[cellID]
end

@define_boundary Symmetry Divergence{Linear} VectorField begin
    ap = term.sign*term.flux[fID]
    _tangential_divergence(ap, term.phi[cellID], face.normal, component)
end

@define_boundary Symmetry Divergence{Upwind} VectorField begin
    ap = term.sign*term.flux[fID]
    _tangential_divergence(ap, term.phi[cellID], face.normal, component)
end

@define_boundary Symmetry Divergence{LUST} VectorField begin
    ap = term.sign*term.flux[fID]
    _tangential_divergence(ap, term.phi[cellID], face.normal, component)
end

# Scalars cancel exactly. A vector leaves the difference between its tangential
# face projection and its owner-cell value.
@define_boundary Symmetry Divergence{BoundedUpwind} ScalarField begin
    0.0, 0.0
end

@define_boundary Symmetry Divergence{BoundedUpwind} VectorField begin
    (; normal) = face
    ap = term.sign*term.flux[fID]
    vc = term.phi[cellID]
    vn = (vc⋅normal)*normal
    0.0, ap*vn[component.value]
end

@define_boundary Symmetry Si begin
    0.0, 0.0
end
