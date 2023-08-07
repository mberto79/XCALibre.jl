export scheme!, scheme_source!

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
    b, nzval, cell, cID)  where {F,P,I} = begin
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
    b, nzval, cell, cID) where {F,P,I} = begin
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
    b, nzval, cell, cID)  where {F,P,I} = begin
    ap = term.sign*(term.flux[cID] * cell.volume)
    nzval[cID] += ap
    nothing
end