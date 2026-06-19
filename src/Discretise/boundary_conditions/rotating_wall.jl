export RotatingWall

"""
    RotatingWall <: AbstractPhysicalConstraint

RotatingWall boundary condition model for no-slip  for rotating RotatingWalls (rotating around an axis). It has been strictly implemented to work with vectors only. 

# Inputs
- `rpm` provides the rotating speed (internally converted to radian/second)
- `centre` vector indicating the location of the rotation centre e.g. [0,0,0]
- `axis` unit vector indicating the rotation axis e.g. [1,0,0]

# Example
    RotatingWall(:inner_wall, rpm=100, centre=[0,0,0], axis=[0,0,1])
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
    
    vb = bc.value(face) # call functor stored in "value"
    vc = phi[cellID]
    vc_n = (vc⋅normal)*normal
    vb_n = (vb⋅normal)*normal
    vb_p = (vb - vb_n) # parallel component of given boundary vector
   
    ap, ap*(vb_p[component.value] + vc_n[component.value])
end

@define_boundary RotatingWall Laplacian{Linear} ScalarField begin
    0.0, 0.0
end

@define_boundary RotatingWall Divergence{Linear} VectorField begin
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