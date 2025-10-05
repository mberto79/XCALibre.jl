export RotatingWall

"""
    RotatingWall <: AbstractDirichlet

RotatingWall boundary condition model for no-slip  for rotating RotatingWalls (rotating around an axis). It has been strictly implemented to work with vectors only. 

# Inputs
- `ID` represents the name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` should be given as a vector for the velocity e.g. [10,0,0]. For scalar fields such as the pressure the value entry can be omitted or set to zero explicitly.

# Examples
    RotatingWall(:plate, [0, 0, 0]) # no-slip RotatingWall condition for velocity
    RotatingWall(:plate) # corresponding definition for scalars, e.g. pressure
    RotatingWall(:plate, 0) # alternative definition for scalars, e.g. pressure
"""
struct RotatingWall{I,V,R<:UnitRange} <: AbstractPhysicalConstraint
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure RotatingWall

@kwdef struct RotatingWallValue{V,F}
    centre::V 
    axis::V 
    rpm::F 
end

@inline (value::RotatingWallValue)(face) = begin
    (; centre, axis, rpm) = value 
    omega = 2π*rpm/60 
    velocity = -omega*(face.centre - centre) × axis # /norm(axis)
    return velocity
end

adapt_value(value::RotatingWallValue, mesh) = begin
    F = _get_float(mesh)
    (; centre, axis, rpm) = value 
    @assert length(centre) == 3 "Centre must be provided as a vector e.g. [0,0,0]"
    @assert length(axis) == 3 "axis must be provided as a vector e.g. [1,0,0]"
    centre = SVector{3,F}(centre)
    axis = SVector{3,F}(axis)
    RotatingWallValue(centre=centre, axis=axis, rpm=F(rpm))
end

RotatingWall(name::Symbol; centre, axis, rpm) = begin
    RotatingWall(name, RotatingWallValue(centre, axis, rpm))
end

@define_boundary RotatingWall Laplacian{Linear} VectorField begin
    (; area, delta, normal) = face 
    phi = term.phi 
    J = term.flux[fID]
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    # vb = SVector{3}(0.0,0.0,0.0) # do not hard-code in next version
    vb = bc.value(face) # boundary value
    vc = phi[cellID]
    vc_n = (vc⋅normal)*normal
    vb_n = (vb⋅normal)*normal
    vb_p = (vb - vb_n) # parallel component of given boundary vector
   
    ap, ap*(vb_p[component.value] + vc_n[component.value])
end

@define_boundary RotatingWall Laplacian{Linear} ScalarField begin
    # phi = term.phi 
    # values = get_values(phi, component)
    # J = term.flux[fID]
    # (; area, delta) = face 
    # flux = -J*area/delta
    # ap = term.sign*(flux)
    # ap, ap*values[cellID] # original
    0.0, 0.0 # try this
    # 0.0, -flux*delta*bc.value # draft implementation to test!
end

# To-do: Add scala scalar variants of RotatingWall BC in next version (currently using Neumann)

@define_boundary RotatingWall Divergence{Linear} VectorField begin # To-do refactor this code for reusability
    0.0, 0.0
end

@define_boundary RotatingWall Divergence{Upwind} VectorField begin
    0.0, 0.0
end

@define_boundary RotatingWall Divergence{LUST} VectorField begin
    0.0, 0.0
end

@define_boundary RotatingWall Divergence{BoundedUpwind} VectorField begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    -flux, 0.0
end

# # Scalar implementations for divergence operator
# @define_boundary RotatingWall Divergence{Upwind} ScalarField begin
#     flux = term.flux[fID]
#     ap = term.sign*(flux) 
#     ap, 0.0 # original

#     # phi = term.phi 
#     # values = get_values(phi, component)
#     # 0.0, -ap*values[cellID] # try this
# end

# @define_boundary RotatingWall Divergence{Linear} ScalarField begin
#     flux = term.flux[fID]
#     ap = term.sign*(flux) 
#     ap, 0.0 # original

#     # phi = term.phi 
#     # values = get_values(phi, component)
#     # 0.0, -ap*values[cellID] # try this
# end

# @define_boundary RotatingWall Divergence{LUST} ScalarField begin
#     flux = term.flux[fID]
#     ap = term.sign*(flux) 
#     ap, 0.0 # original

#     # phi = term.phi 
#     # values = get_values(phi, component)
#     # 0.0, -ap*values[cellID] # try this
# end

# # @define_boundary Symmetry Divergence{BoundedUpwind} begin
# #     0.0, 0.0
# # end

# @define_boundary RotatingWall Si begin
#     0.0, 0.0
# end