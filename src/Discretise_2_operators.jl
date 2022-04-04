export Laplacian, Divergence
export aP!, aN!, b!
export scheme!, scheme_source!
export scheme3!
export scheme4!, scheme_source4!


### OPERATORS AND SCHEMES
struct Source{T} <: AbstractSource
    phi::Float64
    type::T
    label::Symbol
end
Source{Constant}(phi) = Source{Constant}(phi, Constant(), :ConstantSource)


Laplacian{Linear}(J, phi) = Laplacian{Linear}(J, phi, [1])
@inline aP!(term::Laplacian{Linear}, A, cell, face, nsign, cID) = begin
    A[cID, cID] += term.sign[1]*(-term.J * face.area)/face.delta
    nothing
end
@inline aN!(term::Laplacian{Linear}, A, cell, face, nsign, cID, nID) = begin
    A[cID, nID] += term.sign[1]*(term.J * face.area)/face.delta
    nothing
end
@inline b!(term::Laplacian{Linear}, b, cell, cID) = begin
    b[cID] = 0.0
    nothing
end

Divergence{Linear}(J, phi) = Divergence{Linear}(J, phi, [1])
@inline aP!(term::Divergence{Linear}, A, cell, face, nsign, cID) = begin
    A[cID, cID] += term.sign[1]*(term.J⋅face.normal*nsign*face.area)/2.0
    nothing
end
@inline aN!(term::Divergence{Linear}, A, cell, face, nsign, cID, nID) = begin
    A[cID, nID] += term.sign[1]*(term.J⋅face.normal*nsign*face.area)/2.0
    nothing
end
@inline b!(term::Divergence{Linear}, b, cell, cID) = begin
    b[cID] = 0.0
    nothing
end

# Implementation as single functions
@inline function scheme!(term::Laplacian{Linear}, A, cell, face, ns, cID, nID)
    ap = term.sign[1]*(-term.J * face.area)/face.delta
    A[cID, cID] += ap
    A[cID, nID] += -ap
end
@inline scheme_source!(term::Laplacian{Linear}, b, cell, cID) = begin
    b[cID] += 0.0
end

@inline function scheme!(term::Divergence{Linear}, A, cell, face, ns, cID, nID)
    ap = term.sign[1]*(term.J⋅face.normal*ns*face.area)/2.0
    A[cID, cID] += ap
    A[cID, nID] += ap
end
@inline scheme_source!(term::Divergence{Linear}, b, cell, cID) = begin
    b[cID] += 0.0
end

# Face-based implementation
@inline function scheme3!(
    term::Laplacian{Linear}, A, b, face, cells, fID, cID1, cID2
    )
    ap = term.sign[1]*(-term.J * face.area)/face.delta
    # A[cID, cID] += ap
    # A[cID, nID] += -ap

    # cell1
    A[cID1,cID1] += ap
    A[cID1,cID2] += -ap
    # b[cID1] += zero(0.0)
    # cell2
    A[cID2,cID2] += ap
    A[cID2,cID1] += -ap
    # b[cID2] += zero(0.0)
end

@inline function scheme3!(
    term::Divergence{Linear}, A, b, face, cells, fID, cID1, cID2
    )
    ap = term.sign[1]*(term.J⋅face.normal*face.area)*0.5
    # A[cID, cID] += ap
    # A[cID, nID] += ap

    # cell1
    # fi = findfirst(isequal(fID), cell1.facesID)
    # nsign = 1.0 #cell1.nsign[fi]
    A[cID1,cID1] += ap
    A[cID1,cID2] += ap
    # b[cID1] += 0.0 #zero(0.0)
    # cell2
    # fi = findfirst(isequal(fID), cell2.facesID)
    # nsign = -1.0 #cell2.nsign[fi]
    ap2 = -ap #*nsign
    A[cID2,cID2] += ap2
    A[cID2,cID1] += ap2
    # b[cID2] += 0.0 #zero(0.0)
end

@inline function scheme4!(term::Laplacian{Linear}, nzval, cell, face, ns, cIndex, nIndex)
    ap = term.sign[1]*(-term.J * face.area)/face.delta

    nzval[cIndex] += ap
    nzval[nIndex] += -ap

    # A[cID, cID] += ap
    # A[cID, nID] += -ap
    nothing
end
@inline scheme_source4!(term::Laplacian{Linear}, b, cell, cID) = begin
    b[cID] += 0.0; nothing
end

@inline function scheme4!(term::Divergence{Linear}, nzval, cell, face, ns, cIndex, nIndex)
    ap = term.sign[1]*(term.J⋅face.normal*ns*face.area)/2.0

    nzval[cIndex] += ap
    nzval[nIndex] += ap

    # A[cID, cID] += ap
    # A[cID, nID] += ap
    nothing
end
@inline scheme_source4!(term::Divergence{Linear}, b, cell, cID) = begin
    b[cID] += 0.0; nothing
end