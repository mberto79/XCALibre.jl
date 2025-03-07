export Slip

"""
    Slip <: AbstractDirichlet

Slip boundary condition model for no-slip Slip condition.

### Fields
- 'ID' -- Boundary ID
"""
struct Slip{I,V} <: AbstractPhysicalConstraint
    ID::I
    value::V
end
Adapt.@adapt_structure Slip


function fixedValue(BC::Slip, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Slip{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Slip{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end


@define_boundary Slip Laplacian{Linear} begin
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

# To-do: Add scala scalar variants of Slip BC in next version (currently using Neumann)

@define_boundary Slip Divergence{Linear} begin # To-do refactor this code for reusability
    0.0, 0.0
end

@define_boundary Slip Divergence{Upwind} begin
    0.0, 0.0
end

@define_boundary Slip Divergence{LUST} begin
    0.0, 0.0
end

@define_boundary Slip Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    -flux, 0.0
end