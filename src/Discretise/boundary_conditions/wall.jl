export Wall

"""
    Wall <: AbstractDirichlet

Wall boundary condition model for no-slip  or moving walls (linear motion). It should be applied to the velocity vector, and in most cases, its scalar variant should be applied to scalars.

# Inputs
- `ID` represents the name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` should be given as a vector for the velocity e.g. [10,0,0]. For scalar fields such as the pressure the value entry can be omitted or set to zero explicitly.

# Examples
    Wall(:plate, [0, 0, 0]) # no-slip wall condition for velocity
    Wall(:plate) # corresponding definition for scalars, e.g. pressure
    Wall(:plate, 0) # alternative definition for scalars, e.g. pressure
"""
struct Wall{I,V,R<:UnitRange} <: AbstractPhysicalConstraint
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Wall

Wall(name::Symbol) = Wall(name, 0)

@define_boundary Wall Laplacian{Linear} VectorField begin
    (; area, delta, normal) = face 
    phi = term.phi 
    J = term.flux[fID]
    flux = J*area/delta
    ap = (-flux)
    
    # vb = SVector{3}(0.0,0.0,0.0) # do not hard-code in next version
    vb = bc.value # boundary value
    vc = phi[cellID]
    vc_n = (vc⋅normal)*normal
    vb_n = (vb⋅normal)*normal
    vb_p = (vb - vb_n) # parallel component of given boundary vector
   
    ap, ap*(vb_p[component.value] + vc_n[component.value])
end

@define_boundary Wall Laplacian{Linear} ScalarField begin
    # phi = term.phi 
    # values = get_values(phi, component)
    # J = term.flux[fID]
    # (; area, delta) = face 
    # flux = -J*area/delta
    # ap = (flux)
    # ap, ap*values[cellID] # original
    0.0, 0.0 # try this
    # 0.0, -flux*delta*bc.value # draft implementation to test!
end

# To-do: Add scala scalar variants of Wall BC in next version (currently using Neumann)

@define_boundary Wall Divergence{Linear} VectorField begin # To-do refactor this code for reusability
    0.0, 0.0
end

@define_boundary Wall Divergence{Upwind} VectorField begin
    0.0, 0.0
end

@define_boundary Wall Divergence{LUST} VectorField begin
    0.0, 0.0
end

@define_boundary Wall Divergence{BoundedUpwind} VectorField begin
    flux = term.flux[fID]
    ap = (flux)
    -flux, 0.0
end

# Scalar implementations for divergence operator
@define_boundary Wall Divergence{Upwind} ScalarField begin
    flux = term.flux[fID]
    ap = (flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Wall Divergence{Linear} ScalarField begin
    flux = term.flux[fID]
    ap = (flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Wall Divergence{LUST} ScalarField begin
    flux = term.flux[fID]
    ap = (flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

# @define_boundary Symmetry Divergence{BoundedUpwind} begin
#     0.0, 0.0
# end

@define_boundary Wall Si begin
    0.0, 0.0
end