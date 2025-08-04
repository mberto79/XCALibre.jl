export Extrapolated


"""
    Extrapolated <: AbstractNeumann

This boundary condition extrapolates the face value using the interior cell value. Equivalent to setting a zero gradient boundary condition (semi-implicitly). It applies to both scalar and vector fields.

# Example
    Extrapolated(:outlet)
"""
struct Extrapolated{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Extrapolated

Extrapolated(name::Symbol) = Extrapolated(name , 0)

@define_boundary Extrapolated Laplacian{Linear} begin
    # For now this is hard-coded as zero-gradient. To-do extension to any input gradient
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = (flux)
    ap, ap*values[cellID] # original
    # 0.0, 0.0 # try this
    # 0.0, -flux*bc.value # draft implementation to test!
end

@define_boundary Extrapolated Divergence{Linear} begin
    flux = term.flux[fID]
    ap = (flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Extrapolated Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = (flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Extrapolated Divergence{LUST} begin
    flux = term.flux[fID]
    ap = (flux) 
    ap, 0.0 # original

    # phi = term.phi 
    # values = get_values(phi, component)
    # 0.0, -ap*values[cellID] # try this
end

@define_boundary Extrapolated Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = (flux)
    ap-flux, 0.0
end

@define_boundary Extrapolated Si begin
    0.0, 0.0
end