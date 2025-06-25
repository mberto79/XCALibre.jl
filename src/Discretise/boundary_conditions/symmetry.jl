export Symmetry

"""
    Symmetry <: AbstractBoundary

Symmetry boundary condition vector and scalar fields. Notice that for scalar fields, this boundary condition applies an explicit zero gradient condition. In some rare cases, the use of an `Extrapolated` condition for scalars may be beneficial (to assign a semi-implicit zero gradient condition)

# Input
- `ID` Name of the boundary given as a symbol (e.g. :freestream). Internally it gets replaced with the boundary index ID

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
    phi = term.phi 
    J = term.flux[fID]
    # flux = 2.0*J*area/delta # previous
    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    vp = vc - vn
    ap, ap*vp[component.value]

    # ac, an = _symmetry_normal_stress(component, vc, flux, normal)
    # ac, an
end

# _symmetry_normal_stress(component::XDir, vc, flux, n) = begin
#     ac = flux*n[1]^2
#     an = -flux*n[1]*(vc[2]*n[2] + vc[3]*n[3])
#     ac, an
# end 

# _symmetry_normal_stress(component::YDir, vc, flux, n) = begin
#     ac = flux*n[2]^2
#     an = -flux*n[2]*(vc[1]*n[1] + vc[3]*n[3])
#     ac, an
# end 

# _symmetry_normal_stress(component::ZDir, vc, flux, n) = begin
#     ac = flux*n[3]^2
#     an = -flux*n[3]*(vc[1]*n[1] + vc[2]*n[2])
#     ac, an
# end 

@define_boundary Symmetry Laplacian{Linear} ScalarField begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
    # ap, ap*values[cellID] # original
    0.0, 0.0 # go for this!
end

# To-do: Add scalar variants of Wall BC in next version (currently using Neumann)

@define_boundary Symmetry Divergence{Linear} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

@define_boundary Symmetry Divergence{Upwind} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

@define_boundary Symmetry Divergence{LUST} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

@define_boundary Symmetry Divergence{Linear} VectorField begin
    # 0.0, 0.0

    (; normal) = face 
    phi = term.phi
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    # vp = vc - vn
    ap, ap*vn[component.value]
    # 0.0, ap*(vc[component.value] - vn[component.value])
end

@define_boundary Symmetry Divergence{Upwind} VectorField begin
    # 0.0, 0.0

    (; normal) = face 
    phi = term.phi
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    # vp = vc - vn
    ap, ap*vn[component.value]
    # 0.0, ap*(vc[component.value] - vn[component.value])
end

@define_boundary Symmetry Divergence{LUST} VectorField begin
    # 0.0, 0.0

    (; normal) = face 
    phi = term.phi
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    # vp = vc - vn
    ap, ap*vn[component.value]
    # 0.0, ap*(vc[component.value] - vn[component.value])
end

@define_boundary Symmetry Divergence{BoundedUpwind} begin
    0.0, 0.0
end

@define_boundary Symmetry Si begin
    0.0, 0.0
end