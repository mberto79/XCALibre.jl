export scheme!, scheme_source!

# TIME 
@inline function scheme!(
    term::Operator{F,P,I,Time{Steady}}, 
    nzval, cell, face,  cellN, ns, cIndex, nIndex, fID)  where {F,P,I}
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{Steady}}, 
    b, nzval, cell, cID, cIndex)  where {F,P,I} = begin
    nothing
end

# LAPLACIAN

@inline function scheme!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    nzval, cell, face,  cellN, ns, cIndex, nIndex, fID)  where {F,P,I}
    ap = term.sign*(-term.flux[fID] * face.area)/face.delta
    nzval[cIndex] += ap
    nzval[nIndex] += -ap
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    b, nzval, cell, cID, cIndex)  where {F,P,I} = begin
    nothing
end

# DIVERGENCE

@inline function scheme!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    nzval, cell, face, cellN, ns, cIndex, nIndex, fID)  where {F,P,I}
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    weight = norm(xf - xC)/norm(xN - xC)
    one_minus_weight = one(eltype(weight)) - weight
    ap = term.sign*(term.flux[fID]*ns)
    nzval[cIndex] += ap*one_minus_weight
    nzval[nIndex] += ap*weight
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    b, nzval, cell, cID, cIndex) where {F,P,I} = begin
    nothing
end

@inline function scheme!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    nzval, cell, face, cellN, ns, cIndex, nIndex, fID)  where {F,P,I}
    ap = term.sign*(term.flux[fID]*ns)
    nzval[cIndex] += max(ap, 0.0)
    nzval[nIndex] += -max(-ap, 0.0)
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    b, nzval, cell, cID, cIndex) where {F,P,I} = begin
    nothing
end

# IMPLICIT SOURCE

@inline function scheme!(
    term::Operator{F,P,I,Si}, 
    nzval, cell, face,  cellN, ns, cIndex, nIndex, fID)  where {F,P,I}
    # ap = term.sign*(-term.flux[cIndex] * cell.volume)
    # nzval[cIndex] += ap
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Si}, 
    b, nzval, cell, cID, cIndex)  where {F,P,I} = begin
    phi = term.phi
    # ap = max(flux, 0.0)
    # ab = min(flux, 0.0)*phi[cID]
    flux = term.sign*term.flux[cID]*cell.volume # indexed with cID
    nzval[cIndex] += flux # indexed with cIndex
    # flux = term.sign*term.flux[cID]*cell.volume*phi[cID]
    # b[cID] -= flux
    nothing
end