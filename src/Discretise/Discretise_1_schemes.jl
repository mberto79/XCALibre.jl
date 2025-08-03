export scheme!, scheme_source!, source!

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b" 
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

# TIME 

# SteadyState
@inline function scheme!(
    term::Time{SteadyState,F,P}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P}
    # nothing
    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Time{SteadyState,F,P}, cell, cID, cIndex, prev, runtime
    )  where {F,P<:ScalarField} = begin
    0.0, 0.0
end

@inline scheme_source!(
    term::Time{SteadyState,F,P}, cell, cID, cIndex, prev, runtime
    )  where {F,P<:VectorField} = begin
    0.0, 0.0, 0.0, 0.0
end

## Euler
@inline function scheme!(
    term::Time{Euler,F,P}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P}
    # nothing
    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Time{Euler,F,P}, cell, cID, cIndex, prev, runtime)  where {F,P<:ScalarField} = begin
        volume = cell.volume
        vol_rdt = volume/runtime.dt
        
        # Increment sparse and b arrays 
        ac = vol_rdt
        b = prev[cID]*vol_rdt
        return ac, b
end
@inline scheme_source!(
    term::Time{Euler,F,P}, cell, cID, cIndex, prev, runtime)  where {F,P<:VectorField} = begin
        volume = cell.volume
        vol_rdt = volume/runtime.dt
        
        # Increment sparse and b arrays 
        ac = vol_rdt
        b = prev[cID]*vol_rdt
        return ac, b[1], b[2], b[3]
end

# LAPLACIAN
@inline function scheme!(
    term::Laplacian{Linear,F,P}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P}

    
    (; area, normal, delta, e) = face
    dPN = cellN.centre - cell.centre
    n = ns*normal
    Ef = dPN*(norm(n)^2/(dPN⋅n))*area

    # Sf = ns*area*normal # original
    # e = ns*e # original
    # Ef = ((Sf⋅Sf)/(Sf⋅e))*e # original
    Ef_mag = norm(Ef)
    ap = (term.flux[fID] * Ef_mag)/delta

    # ap = (term.flux[fID] * area)/delta
    
    # Increment sparse array
    ac = -ap
    an = ap
    return ac, an
end
@inline scheme_source!(term::Laplacian{Linear,F,P}, cell, cID, cIndex, prev, runtime
)  where {F,P<:ScalarField} = begin
    0.0, 0.0
end
@inline scheme_source!(term::Laplacian{Linear,F,P}, cell, cID, cIndex, prev, runtime
)  where {F,P<:VectorField} = begin
    0.0, 0.0, 0.0, 0.0
end

# DIVERGENCE

# Linear
@inline function scheme!(
    term::Divergence{Linear,F,P}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P}
    # Retrieve mesh centre values
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    
    # Calculate weights using normal functions
    # weight = norm(xN - xf)/norm(xN - xC)
    weight = norm(xN - xf)/(norm(xN - xf) + norm(xC - xf))
    one_minus_weight = one(eltype(weight)) - weight

    # Calculate required increment
    ap = (term.flux[fID]*ns)
    ac = ap*one_minus_weight
    an = ap*weight
    return ac, an
end
@inline scheme_source!(term::Divergence{Linear,F,P}, cell, cID, cIndex, prev, runtime
) where {F,P<:ScalarField} = begin
    0.0, 0.0
end
@inline scheme_source!(term::Divergence{Linear,F,P}, cell, cID, cIndex, prev, runtime
) where {F,P<:VectorField} = begin
    0.0, 0.0, 0.0, 0.0
end

# Upwind
@inline function scheme!(
    term::Divergence{Upwind,F,P}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P}
    # Calculate required increment
    ap = (term.flux[fID]*ns)
    ac = max(ap, 0.0) 
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(term::Divergence{Upwind,F,P}, cell, cID, cIndex, prev, runtime
) where {F,P<:ScalarField} = begin
    0.0, 0.0
end
@inline scheme_source!(term::Divergence{Upwind,F,P}, cell, cID, cIndex, prev, runtime
) where {F,P<:VectorField} = begin
    0.0, 0.0, 0.0, 0.0
end

# LUST
@inline function scheme!(
    term::Divergence{LUST,F,P}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P}
    # Retrieve mesh centre values
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    
    # Calculate weights using normal functions
    weight = norm(xN - xf)/(norm(xN - xf) + norm(xC - xf))
    one_minus_weight = one(eltype(weight)) - weight

    # Calculate coefficients
    ap = (term.flux[fID]*ns)
    acLinear = ap*one_minus_weight
    anLinear = ap*weight
    acUpwind = max(ap, 0.0) 
    anUpwind = -max(-ap, 0.0)
    ac = 0.75*acLinear + 0.25*acUpwind
    an = 0.75*anLinear + 0.25*anUpwind
    return ac, an
end
@inline scheme_source!(term::Divergence{LUST,F,P}, cell, cID, cIndex, prev, runtime
) where {F,P<:ScalarField} = begin
    0.0, 0.0
end
@inline scheme_source!(term::Divergence{LUST,F,P}, cell, cID, cIndex, prev, runtime
) where {F,P<:VectorField} = begin
    0.0, 0.0
end

# BoundedUpwind
@inline function scheme!(
    term::Divergence{BoundedUpwind,F,P}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P}
    # Calculate required increment
    volume = cell.volume
    ap = (term.flux[fID]*ns)
    ac = max(ap, 0.0) - term.flux[fID]#*volume 
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(
    term::Divergence{BoundedUpwind,F,P}, cell, cID, cIndex, prev, runtime
    ) where {F,P<:ScalarField} = begin
    0.0, 0.0
end
@inline scheme_source!(
    term::Divergence{BoundedUpwind,F,P}, cell, cID, cIndex, prev, runtime
    ) where {F,P<:VectorField} = begin
    0.0, 0.0
end

# IMPLICIT SOURCE
@inline function scheme!(
    term::Si{S,F,P}, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    ) where {S,F,P}
    0.0, 0.0
end
@inline scheme_source!(
    term::Si{S,F,P}, cell, cID, cIndex, prev, runtime) where {S,F,P<:ScalarField} = begin
    # Retrieve and calculate flux for cell 
    flux = term.flux[cID]*cell.volume # indexed with cID
    ac = flux # indexed with cIndex
    ac, 0.0
end
@inline scheme_source!(
    term::Si{S,F,P}, cell, cID, cIndex, prev, runtime) where {S,F,P<:VectorField} = begin
    # Retrieve and calculate flux for cell 
    flux = term.flux[cID]*cell.volume # indexed with cID
    ac = flux # indexed with cIndex
    ac, 0.0, 0.0, 0.0
end

# Explicit source

source!(term::Source{P}, volume, cID) where {P<:ScalarField} = begin
    term.field[cID]*volume
end

source!(term::Source{P}, volume, cID) where {P<:VectorField} = begin
    src = term.field[cID]*volume
    src[1], src[2], src[3]
end
