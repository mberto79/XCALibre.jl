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

## Crank-Nicholson
@inline function scheme!(
    term::Operator{F,P,I,Time{CrankNicolson}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    # nothing
    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{CrankNicolson}}, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
        volume = cell.volume
        vol_rdt = volume/runtime.dt # advance solution by dt/2 only
        
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
    ## Potential simplified form for performance, needs checking before use in release
    # dPN = cellN.centre - cell.centre
    # n = ns*normal
    # Ef = dPN*(norm(n)^2/(dPN⋅n))*area # this works 
    # Ef = dPN*(one(typeof(ns))/(dPN⋅n))*area # a little faster but a few more iter

    # Use form below to ensure correctness, could be simplified for performance
    Sf = ns*area*normal # original
    e = ns*e # original
    Ef = ((Sf⋅Sf)/(Sf⋅e))*e # original
    Ef_mag = norm(Ef)
    ap = term.sign*(term.flux[fID]*Ef_mag)/delta

    # ap = term.sign*(term.flux[fID]*area)/delta # Initial form used
    
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
    f = face.centre
    C = cell.centre
    N = cellN.centre

    # calculate distance vectors
    d_fC = C - f 
    d_fN = N - f
    
    # Calculate weights using normal functions
    weight = norm(d_fN)/(norm(d_fC) + norm(d_fN))
    one_minus_weight = one(eltype(weight)) - weight

    # Calculate required increment
    ap = term.sign*(term.flux[fID]*ns)
    # ac = ap*one_minus_weight
    # an = ap*weight
    ac = ap*weight
    an = ap*one_minus_weight
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

# LUST
@inline function scheme!(
    term::Operator{F,P,I,Divergence{LUST}}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Retrieve mesh centre values
    f = face.centre
    C = cell.centre
    N = cellN.centre

    # calculate distance vectors
    d_fC = C - f 
    d_fN = N - f
    
    # Calculate weights using normal functions
    weight = norm(d_fN)/(norm(d_fC) + norm(d_fN))
    one_minus_weight = one(eltype(weight)) - weight

    # Calculate coefficients
    ap = term.sign*(term.flux[fID]*ns)
    # acLinear = ap*one_minus_weight
    # anLinear = ap*weight
    acLinear = ap*weight 
    anLinear = ap*one_minus_weight
    acUpwind = max(ap, 0.0) 
    anUpwind = -max(-ap, 0.0)
    ac = 0.75*acLinear + 0.25*acUpwind
    an = 0.75*anLinear + 0.25*anUpwind
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{LUST}}, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
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
