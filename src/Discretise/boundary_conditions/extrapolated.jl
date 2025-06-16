export Extrapolated


"""
    Extrapolated <: AbstractNeumann

This boundary condition extrapolates the face value using the interior cell value. 

### Fields
- 'ID' -- Boundary ID
- `value` -- Scalar or Vector value for Extrapolated boundary condition.
"""
struct Extrapolated{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Extrapolated

Extrapolated(name::Symbol) = Extrapolated(name , 0)

"""
    fixedValue(BC::AbstractNeumann, ID::I, value::V) where {I<:Integer,V}

"""
# function fixedValue(BC::AbstractNeumann, ID::I, value::V) where {I<:Integer,V}
#     # Exception 1: value is scalar
#     if V <: Number
#         return Extrapolated{I,eltype(value)}(ID, value)
#     # Exception 2: value is vector
#     elseif V <: Vector
#         if length(value) == 3 
#             nvalue = SVector{3, eltype(value)}(value)
#             return Extrapolated{I,typeof(nvalue)}(ID, nvalue)
#         # Error statement if vector is invalid        
#         else
#             throw("Only vectors with three components can be used")
#         end
#     # Error if value is not scalar or vector
#     else
#         throw("The value provided should be a scalar or a vector")
#     end
# end

@define_boundary Extrapolated Laplacian{Linear} begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
    ap, ap*values[cellID] # original
    # 0.0, 0.0 # try this
    # 0.0, -flux*bc.value # draft implementation to test!
end

@define_boundary Extrapolated Divergence{Linear} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Extrapolated Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Extrapolated Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Extrapolated Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

@define_boundary Extrapolated Si begin
    0.0, 0.0
end