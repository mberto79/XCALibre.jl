export Wall

"""
    Wall <: AbstractDirichlet

Wall boundary condition model for no-slip wall condition.

### Fields
- 'ID' -- Boundary ID
"""
struct Wall{I,V} <: AbstractPhysicalConstraint
    ID::I
    value::V
end
Adapt.@adapt_structure Wall


function fixedValue(BC::Wall, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Wall{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Wall{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end


@define_boundary Wall Laplacian{Linear} VectorField begin
    (; area, delta, normal) = face 
    phi = term.phi 
    J = term.flux[fID]
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    vb = SVector{3}(0.0,0.0,0.0) # do not hard-code in next version
    vc = phi[cellID]
    vc_n = (vc⋅normal)*normal
    vb_n = (vb⋅normal)*normal
    vb_p = (vb - vb_n) # parallel component of given boundary vector
   
    ap, ap*(vb_p[component.value] + vc_n[component.value])
end

@define_boundary Wall Laplacian{Linear} ScalarField begin
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
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
    ap = term.sign*(flux)
    -flux, 0.0
end

# Scalar implementations for divergence operator
@define_boundary Wall Divergence{Upwind} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Wall Divergence{Linear} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Wall Divergence{LUST} ScalarField begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end