export Empty

#= NOTE: 
Interpolation for Empty boundaries is implemented as Neumann (zero gradient) for now.

This will probably need changing once a dedicated ZeroGradient BC type is implemented
=#

"""
    Empty <: AbstractEmpty

Empty boundary condition model *(currently only configured for zero gradient)*

### Fields
- 'ID' -- Boundary ID
- `value` -- Scalar or Vector value for Empty boundary condition.
"""
struct Empty{I,V} <: AbstractEmpty
    ID::I 
    value::V 
end
Adapt.@adapt_structure Empty

Empty(patch::Symbol) = Empty(patch, 0)

function fixedValue(BC::AbstractEmpty, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: value is scalar
    if V <: Number
        return Empty{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Empty{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid        
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

@define_boundary Empty Laplacian{Linear} begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
    ap, ap*values[cellID] # original
    # 0.0, 0.0 # try this
    # 0.0, -flux*delta*bc.value # draft implementation to test!
end

@define_boundary Empty Divergence{Linear} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Empty Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Empty Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Empty Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

@define_boundary Empty Si begin
    0.0, 0.0
end