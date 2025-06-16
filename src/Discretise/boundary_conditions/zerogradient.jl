export Zerogradient


"""
    Zerogradient <: AbstractNeumann

Zerogradient boundary condition model *(currently only configured for zero gradient)*

### Fields
- 'ID' -- Boundary ID
- `value` -- Scalar or Vector value for Zerogradient boundary condition.
"""
struct Zerogradient{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Zerogradient

Zerogradient(name::Symbol) = Zerogradient(name , 0)


function fixedValue(BC::Zerogradient, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: value is scalar
    if V <: Number
        return Zerogradient{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Zerogradient{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid        
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

@define_boundary Zerogradient Laplacian{Linear} begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    # phi = term.phi 
    # values = get_values(phi, component)
    # J = term.flux[fID]
    # (; area, delta) = face 
    # # flux = -J*area/delta
    # flux = -J*area # /delta
    # ap = term.sign*(flux)
    # # ap, ap*values[cellID] # original
    0.0, 0.0 # try this
end

@define_boundary Zerogradient Divergence{Linear} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Zerogradient Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Zerogradient Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Zerogradient Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

@define_boundary Zerogradient Si begin
    0.0, 0.0
end