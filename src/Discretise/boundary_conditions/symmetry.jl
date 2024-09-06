export Symmetry

"""
    Symmetry <: AbstractBoundary

Symmetry boundary condition model for velocity and scalar field.

### Fields
- 'ID' -- Boundary ID
"""
struct Symmetry{I,V} <: AbstractBoundary
    ID::I
    value::V
end
Adapt.@adapt_structure Symmetry


function fixedValue(BC::Symmetry, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Symmetry{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Symmetry{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

@define_boundary Symmetry Laplacian{Linear} begin
    (; area, delta, normal) = face 
    phi = term.phi 
    J = term.flux[fID]
    flux = 2*J*area/delta
    ap = term.sign[1]*(-flux)

    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    vp = vc - vn
    ap, ap*vp[component.value]
end

# To-do: Add scalar variants of Wall BC in next version (currently using Neumann)

@define_boundary Symmetry Divergence{Linear} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end

@define_boundary Symmetry Divergence{Upwind} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end

@define_boundary Symmetry Divergence{LUST} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end