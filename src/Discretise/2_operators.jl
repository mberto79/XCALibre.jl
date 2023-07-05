export Laplacian, Divergence
export aP!, aN!, b!
export scheme!, scheme_source!
export scheme3!
export scheme4!, scheme_source4!


### OPERATORS AND SCHEMES

# struct Source{T} <: AbstractSource
#     phi::Float64
#     type::T
#     label::Symbol
# end
# Source{Constant}(phi) = Source{Constant}(phi, Constant(), :ConstantSource)


### LAPLACIAN (constant scalar)

# @inline function scheme!(
#     term::Laplacian{Linear}, nzval, cell, face,  cellN, ns, cIndex, nIndex, fID
#     )
#     ap = term.sign[1]*(-term.J * face.area)/face.delta
#     nzval[cIndex] += ap
#     nzval[nIndex] += -ap
#     nothing
# end
# @inline scheme_source!(
#     term::Laplacian{Linear}, b, cell, cID) = begin
#     # b[cID] += 0.0
#     nothing
# end

### LAPLACIAN (non-uniform scalar field)

# Laplacian{Linear}(J::T, phi) where T<:FaceScalarField = begin
#     Laplacian{Linear, T}(J, phi, [1])
# end

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
    b, cell, cID)  where {F,P,I} = begin
    # b[cID] += 0.0
    nothing
end

### DIVERGENCE (Constant vector field)

# @inline function scheme!(
#     term::Divergence{Linear}, nzval, cell, face, cellN, ns, cIndex, nIndex, fID
#     )
#     xf = face.centre
#     xC = cell.centre
#     xN = cellN.centre
#     weight = norm(xf - xC)/norm(xN - xC)
#     ap = term.sign[1]*(term.J⋅face.normal*ns*face.area) # need to implement weights
#     nzval[cIndex] += ap*(1.0 - weight)
#     nzval[nIndex] += ap*weight
#     nothing
# end
# @inline scheme_source!(term::Divergence{Linear}, b, cell, cID) = begin
#     # b[cID] += 0.0
#     nothing
# end

### DIVERGENCE (Non-uniform vector field)

# Divergence{Linear}(J::FaceScalarField{I,F}, phi) where {I,F} = begin
#     Divergence{Linear, FaceScalarField{I,F}}(J, phi, [1])
# end

@inline function scheme!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    nzval, cell, face, cellN, ns, cIndex, nIndex, fID)  where {F,P,I}
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    weight = norm(xf - xC)/norm(xN - xC)
    one_minus_weight = one(eltype(weight)) - weight
    ap = term.sign*(term.flux[fID]*ns)
    # ap = term.sign[1]*(term.J⋅face.normal*ns*face.area)
    nzval[cIndex] += ap*one_minus_weight
    nzval[nIndex] += ap*weight
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    b, cell, cID) where {F,P,I} = begin
    # b[cID] += 0.0
    nothing
end

# Divergence{Linear}(J::FaceVectorField{I,F}, phi) where {I,F}= begin
#     Divergence{Linear, FaceVectorField{I,F}}(J, phi, [1])
# end

# @inline function scheme!(
#     term::Divergence{Linear}, nzval, cell, face, cellN, ns, cIndex, nIndex, fID
#     )  where {I,F}
#     xf = face.centre
#     xC = cell.centre
#     xN = cellN.centre
#     weight = norm(xf - xC)/norm(xN - xC)
#     # ap = term.sign[1]*(term.J(fID)*ns)
#     ap = term.sign[1]*(term.J(fID)⋅face.normal*ns*face.area)
#     nzval[cIndex] += ap*(1.0 - weight)
#     nzval[nIndex] += ap*weight
#     nothing
# end
# @inline scheme_source!(
#     term::Divergence{Linear}, b, cell, cID
#     ) where {I,F} = begin
#     # b[cID] += 0.0
#     nothing
# end