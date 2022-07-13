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


### LAPLACIAN (constant scalar)

Laplacian{Linear}(J::T, phi) where T<:AbstractFloat = begin
    Laplacian{Linear, Float64}(J, phi, [1])
end

@inline function scheme!(
    term::Laplacian{Linear, Float64}, nzval, cell, face,  cellN, ns, cIndex, nIndex, fID
    )
    ap = term.sign[1]*(-term.J * face.area)/face.delta
    nzval[cIndex] += ap
    nzval[nIndex] += -ap
    nothing
end
@inline scheme_source!(
    term::Laplacian{Linear, Float64}, b, cell, cID) = begin
    # b[cID] += 0.0
    nothing
end

### LAPLACIAN (non-uniform scalar field)

Laplacian{Linear}(J::T, phi) where T<:FaceScalarField = begin
    Laplacian{Linear, T}(J, phi, [1])
end

@inline function scheme!(
    term::Laplacian{Linear, T}, nzval, cell, face,  cellN, ns, cIndex, nIndex, fID
    ) where T<:FaceScalarField
    ap = term.sign[1]*(-term.J(fID) * face.area)/face.delta
    nzval[cIndex] += ap
    nzval[nIndex] += -ap
    nothing
end
@inline scheme_source!(
    term::Laplacian{Linear, T}, b, cell, cID
    ) where T<:FaceScalarField= begin
    # b[cID] += 0.0
    nothing
end

### DIVERGENCE (Constant vector field)

Divergence{Linear}(J::Vector{Float64}, phi) = begin
    J_static = SVector{3, Float64}(J)
    Divergence{Linear, SVector{3, Float64}}(J_static, phi, [1])
end

@inline function scheme!(
    term::Divergence{Linear, SVector{3, Float64}}, nzval, cell, face, cellN, ns, cIndex, nIndex, fID
    )
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    weight = norm(xf - xC)/norm(xN - xC)
    ap = term.sign[1]*(term.J⋅face.normal*ns*face.area) # need to implement weights
    nzval[cIndex] += ap*(1.0 - weight)
    nzval[nIndex] += ap*weight
    nothing
end
@inline scheme_source!(term::Divergence{Linear, SVector{3, Float64}}, b, cell, cID) = begin
    # b[cID] += 0.0
    nothing
end

### DIVERGENCE (Non-uniform vector field)

Divergence{Linear}(J::FaceScalarField{I,F}, phi) where {I,F}= begin
    Divergence{Linear, FaceScalarField{I,F}}(J, phi, [1])
end

@inline function scheme!(
    term::Divergence{Linear, FaceScalarField{I,F}}, nzval, cell, face, cellN, ns, cIndex, nIndex, fID
    )  where {I,F}
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    weight = norm(xf - xC)/norm(xN - xC)
    ap = term.sign[1]*(term.J(fID)*ns)
    # ap = term.sign[1]*(term.J⋅face.normal*ns*face.area)
    nzval[cIndex] += ap*(1.0 - weight)
    nzval[nIndex] += ap*weight
    nothing
end
@inline scheme_source!(
    term::Divergence{Linear, FaceScalarField{I,F}}, b, cell, cID
    ) where {I,F} = begin
    # b[cID] += 0.0
    nothing
end

Divergence{Linear}(J::FaceVectorField{I,F}, phi) where {I,F}= begin
    Divergence{Linear, FaceVectorField{I,F}}(J, phi, [1])
end

@inline function scheme!(
    term::Divergence{Linear, FaceVectorField{I,F}}, nzval, cell, face, cellN, ns, cIndex, nIndex, fID
    )  where {I,F}
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    weight = norm(xf - xC)/norm(xN - xC)
    # ap = term.sign[1]*(term.J(fID)*ns)
    ap = term.sign[1]*(term.J(fID)⋅face.normal*ns*face.area)
    nzval[cIndex] += ap*(1.0 - weight)
    nzval[nIndex] += ap*weight
    nothing
end
@inline scheme_source!(
    term::Divergence{Linear, FaceVectorField{I,F}}, b, cell, cID
    ) where {I,F} = begin
    # b[cID] += 0.0
    nothing
end