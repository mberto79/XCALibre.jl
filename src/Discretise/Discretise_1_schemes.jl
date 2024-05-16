export scheme!, scheme_source!

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b" 
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

# TIME 

# Steady
@inline function scheme!(
    term::Operator{F,P,I,Time{Steady}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    # nothing
    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{Steady}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    nothing
end

## Euler
@inline function scheme!(
    term::Operator{F,P,I,Time{Euler}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    # nothing
    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{Euler}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
        # Retrieve cell volume and calculate time step
        # volume = cell.volume
        # rdt = 1/runtime.dt
        volume = cell.volume
        vol_rdt = volume/runtime.dt
        
        # Increment sparse and b arrays 
        # Atomix.@atomic nzval_array[cIndex] += vol_rdt
        # Atomix.@atomic b[cID] += prev[cID]*vol_rdt
        nzval_array[cIndex] += vol_rdt
        b[cID] += prev[cID]*vol_rdt
    nothing
end

# LAPLACIAN

@inline function scheme!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Calculate required increment
    ap = term.sign*(term.flux[fID] * face.area)/face.delta

    # Increment sparse array
    # Atomix.@atomic nzval_array[cIndex] += -ap
    # Atomix.@atomic nzval_array[nIndex] += ap
    # nothing
    ac = -ap
    an = ap
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    nothing
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

    # Increment sparse array
    # Atomix.@atomic nzval_array[cIndex] += ap*one_minus_weight
    # Atomix.@atomic nzval_array[nIndex] += ap*weight
    # nothing
    ac = ap*one_minus_weight
    an = ap*weight
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    nothing
end

# Upwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Calculate required increment
    ap = term.sign*(term.flux[fID]*ns)

    # Increment sparse array only if ap is positive
    # Atomix.@atomic nzval_array[cIndex] += max(ap, 0.0)
    # Atomix.@atomic nzval_array[nIndex] += -max(-ap, 0.0)
    # nothing
    ac = max(ap, 0.0)
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    nothing
end

# IMPLICIT SOURCE
@inline function scheme!(
    term::Operator{F,P,I,Si}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Si}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    
    # Retrieve and calculate flux for cell 
    flux = term.sign*term.flux[cID]*cell.volume # indexed with cID
    
    # Increment sparse array by flux
    # Atomix.@atomic nzval_array[cIndex] += flux # indexed with cIndex
    nzval_array[cIndex] += flux # indexed with cIndex
    nothing
end
