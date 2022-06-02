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


### LAPLACIAN

Laplacian{Linear}(J::Float64, phi) = begin
    Laplacian{Linear, Float64}(J, phi, [1])
end

@inline function scheme!(
    term::Laplacian{Linear, Float64}, nzval, cell, face,  cellN, ns, cIndex, nIndex, fIndex
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


### DIVERGENCE (Constant vector field)

Divergence{Linear}(J::Vector{Float64}, phi) = begin
    J_static = SVector{3, Float64}(J)
    Divergence{Linear, SVector{3, Float64}}(J_static, phi, [1])
end

@inline function scheme!(
    term::Divergence{Linear, SVector{3, Float64}}, nzval, cell, face, cellN, ns, cIndex, nIndex, fIndex
    )
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    weight = norm(xf - xC)/norm(xN - xC)
    ap = term.sign[1]*(term.J⋅face.normal*ns*face.area) # need to implement weights
    nzval[cIndex] += ap*0.5 #(1.0 - weight)
    nzval[nIndex] += ap*0.5 #weight
    nothing
end
@inline scheme_source!(term::Divergence{Linear, SVector{3, Float64}}, b, cell, cID) = begin
    # b[cID] += 0.0
    nothing
end

### DIVERGENCE (Non-uniform vector field)

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
    ap = term.sign[1]*(term.J(fID)⋅face.normal*ns*face.area)
    nzval[cIndex] += ap*0.5 #(1.0 - weight)
    nzval[nIndex] += ap*0.5 #weight
    nothing
end
@inline scheme_source!(
    term::Divergence{Linear, FaceVectorField{I,F}}, b, cell, cID
    ) where {I,F} = begin
    # b[cID] += 0.0
    nothing
end