export Zerogradient


"""
    Zerogradient <: AbstractNeumann

Zerogradient boundary condition model *(explicitly applied to the boundary)*

# Input
- `ID` Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID

# Example
    Zerogradient(:inlet)
"""
struct Zerogradient{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Zerogradient

Zerogradient(name::Symbol) = Zerogradient(name , 0)


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

    ac = max(ap, 0.0)
    su = -min(ap, 0.0) * get_values(term.phi, component)[cellID]

    ac, su
end

@define_boundary Zerogradient Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original

    # # # phi = term.phi 
    # # # values = get_values(phi, component)
    # # # 0.0, -ap*values[cellID] # try this

    # # ac = max(ap, 0.0) 
    # # an = -max(-ap, 0.0)
    # # ac, an*get_values(term.phi, component)[cellID]

    # flux = term.flux[fID]
    # ap = term.sign*(flux) 
    
    # # 1. Flow leaving (ap > 0): 
    # # Use the cell implicitly. Adds to the diagonal, keeping matrix dominant.
    # ac = max(ap, 0.0) 
    
    # # 2. Flow entering (ap < 0): 
    # # Evaluate explicitly to prevent negative diagonals.
    # # We need: -su = ap * phi_P  =>  su = -ap * phi_P
    # su = -min(ap, 0.0) * get_values(term.phi, component)[cellID]
    
    # ac, su
end

@define_boundary Zerogradient Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)

    ac = max(ap, 0.0)
    su = -min(ap, 0.0) * get_values(term.phi, component)[cellID]

    ac, su
end

@define_boundary Zerogradient Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    
    # Internal BoundedUpwind logic: ac = max(-ap, 0), an = -max(-ap, 0)
    # Total Flux = ac*phi_P + an*phi_N
    # For ZeroGradient: phi_N = phi_P
    # Total Flux = (max(-ap, 0) - max(-ap, 0)) * phi_P = 0
    ac = max(-ap, 0.0)
    an = -max(-ap, 0.0)

    ac, -an*get_values(term.phi, component)[cellID]
end

@define_boundary Zerogradient Si begin
    0.0, 0.0
end