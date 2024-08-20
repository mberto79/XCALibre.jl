export scheme!, scheme_source!

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b" 
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

# TIME 

# SteadyState
@inline function scheme!(
    term::Operator{F,P,I,Time{SteadyState}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    # nothing
    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{SteadyState}}, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    0.0, 0.0
end

## Euler
@inline function scheme!(
    term::Operator{F,P,I,Time{Euler}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    # nothing
    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{Euler}}, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
        volume = cell.volume
        vol_rdt = volume/runtime.dt
        
        # Increment sparse and b arrays 
        ac = vol_rdt
        b = prev[cID]*vol_rdt
        return ac, b
end

# LAPLACIAN

@inline function scheme!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}

    (; area, normal, delta, e) = face
    Sf = ns*area*normal
    e = ns*e
    Ef = ((Sf⋅Sf)/(Sf⋅e))*e
    Ef_mag = norm(Ef)
    ap = term.sign*(term.flux[fID] * Ef_mag)/delta

    # ap = term.sign*(term.flux[fID] * area)/delta
    
    # Increment sparse array
    ac = -ap
    an = ap
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Laplacian{Linear}}, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    0.0, 0.0
end

# DIVERGENCE

# Linear
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Retrieve mesh centre values
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    
    # Calculate weights using normal functions
    weight = norm(xf - xC)/norm(xN - xC)
    one_minus_weight = one(eltype(weight)) - weight

    # Calculate required increment
    ap = term.sign*(term.flux[fID]*ns)
    ac = ap*one_minus_weight
    an = ap*weight
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Linear}}, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    0.0, 0.0
end

# Upwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Calculate required increment
    ap = term.sign*(term.flux[fID]*ns)
    ac = max(ap, 0.0) 
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Upwind}}, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    0.0, 0.0
end

# BoundedUpwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Calculate required increment
    volume = cell.volume
    ap = term.sign*(term.flux[fID]*ns)
    ac = max(ap, 0.0) - term.flux[fID]#*volume 
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    0.0, 0.0
end


# IMPLICIT SOURCE
@inline function scheme!(
    term::Operator{F,P,I,Si}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    0.0, 0.0
end
@inline scheme_source!(
    term::Operator{F,P,I,Si}, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    
    # Retrieve and calculate flux for cell 
    flux = term.sign*term.flux[cID]*cell.volume # indexed with cID
    ac = flux # indexed with cIndex
    ac, 0.0
end
