export Dirichlet

"""
    Dirichlet <: AbstractDirichlet

Dirichlet boundary condition model.

# Fields
- 'ID' -- Boundary ID
- `value` -- Scalar or Vector value for Dirichlet boundary condition.
"""
struct Dirichlet{I,V} <: AbstractDirichlet
    ID::I
    value::V
end
Adapt.@adapt_structure Dirichlet

function fixedValue(BC::AbstractDirichlet, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Dirichlet{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Dirichlet{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

@define_boundary Dirichlet Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face 
    flux = J*area/delta
    ap = term.sign*(-flux)
    ap, ap*bc.value
end

@define_boundary Dirichlet Divergence{Linear} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{Upwind} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{LUST} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{BoundedUpwind} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    flux, ap*bc.value
end

@define_boundary Dirichlet Si begin
    0.0, 0.0
end
