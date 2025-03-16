export Symmetry

"""
    Symmetry <: AbstractBoundary

Symmetry boundary condition vector fields. For scalar fields use `Neumann`

### Fields
- 'ID' -- Boundary ID
"""
struct Symmetry{I,V} <: AbstractPhysicalConstraint
    ID::I
    value::V
end
Adapt.@adapt_structure Symmetry

Symmetry(patch::Symbol) = Symmetry(patch, 0)


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

@define_boundary Symmetry Laplacian{Linear} VectorField begin
    (; area, delta, normal) = face 
    phi = term.phi 
    J = term.flux[fID]
    # flux = 2.0*J*area/delta
    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    vp = vc - vn
    ap, ap*vp[component.value]
end

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