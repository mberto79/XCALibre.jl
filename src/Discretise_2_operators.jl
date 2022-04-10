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
@inline function scheme!(term::Laplacian{Linear}, nzval, cell, face, ns, cIndex, nIndex)
    ap = term.sign[1]*(-term.J * face.area)/face.delta
    nzval[cIndex] += ap
    nzval[nIndex] += -ap
    nothing
end
@inline scheme_source!(term::Laplacian{Linear}, b, cell, cID) = begin
    # b[cID] += 0.0
    nothing
end

Divergence{Linear}(J, phi) = Divergence{Linear}(J, phi, [1])
@inline function scheme!(term::Divergence{Linear}, nzval, cell, face, ns, cIndex, nIndex)
    ap = term.sign[1]*(term.J⋅face.normal*ns*face.area*0.5) # need to implement weights
    nzval[cIndex] += ap
    nzval[nIndex] += ap
    nothing
end
@inline scheme_source!(term::Divergence{Linear}, b, cell, cID) = begin
    # b[cID] += 0.0
    nothing
end