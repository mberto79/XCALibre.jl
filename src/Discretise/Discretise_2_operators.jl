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
    term::Laplacian{Linear, Float64}, nzval, cell, face, ns, cIndex, nIndex
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


### DIVERGENCE

Divergence{Linear}(J::Vector{Float64}, phi) = begin
    J_static = SVector{3, Float64}(J)
    Divergence{Linear, SVector{3, Float64}}(J_static, phi, [1])
end

@inline function scheme!(
    term::Divergence{Linear, SVector{3, Float64}}, nzval, cell, face, ns, cIndex, nIndex
    )
    ap = term.sign[1]*(term.J⋅face.normal*ns*face.area*0.5) # need to implement weights
    nzval[cIndex] += ap
    nzval[nIndex] += ap
    nothing
end
@inline scheme_source!(term::Divergence{Linear, SVector{3, Float64}}, b, cell, cID) = begin
    # b[cID] += 0.0
    nothing
end