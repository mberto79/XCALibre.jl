export NeumannFunction

# abstract type XCALibreUserFunctor end

"""
    NeumannFunction(ID, value) <: AbstractNeumann

Neumann boundary condition defined with user-provided function.

# Input
- `ID` Boundary name provided as symbol e.g. :inlet
- `value` Custom function for Neumann boundary condition.

# Function requirements

The function passed to this boundary condition must have the following signature:

    f(coords, time, index) = SVector{3}(ux, uy, uz)

Where, `coords` is a vector containing the coordinates of a face, `time` is the current time in transient simulations (and the iteration number in steady simulations), and `index` is the local face index (from 1 to `N`, where `N` is the number of faces in a given boundary). The function must return an SVector (from StaticArrays.jl) representing the velocity vector. 
"""
struct NeumannFunction{I,V} <: AbstractNeumann
    ID::I 
    value::V 
end
Adapt.@adapt_structure NeumannFunction


function fixedValue(BC::NeumannFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: value is scalar
    if V <: Number
        return NeumannFunction{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return NeumannFunction{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid        
        else
            throw("Only vectors with three components can be used")
        end
    # Exception 3: value is a function
    elseif V <: Function
        return NeumannFunction{I,V}(ID, value)
    # Exception 4: value is a user provided XCALibre functor
    elseif V <: XCALibreUserFunctor
        return NeumannFunction{I,V}(ID, value)
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

@define_boundary NeumannFunction Laplacian{Linear} begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
    ap, ap*values[cellID] # original
    0.0, 0.0 
    # 0.0, -flux*delta*bc.value # draft implementation to test!
end

@define_boundary NeumannFunction Divergence{Linear} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary NeumannFunction Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary NeumannFunction Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary NeumannFunction Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

@define_boundary NeumannFunction Si begin
    0.0, 0.0
end