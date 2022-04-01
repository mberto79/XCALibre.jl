export Laplacian, Divergence
export aP!, aN!, b!

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

# function Laplacian{Linear}(J, phi)
#     ap!(cell, face, nsign, cID) = (-J * face.area)/face.delta
#     an!(cell, face, nsign, cID, nID) = (J * face.area)/face.delta
#     b!(cell, cID) =  0.0
#     Discretisation(phi, [Laplacian{Linear}], [1], ap!, an!, b!)
# end

# function Divergence{Linear}(J, phi)
#     ap!(i) = J*phi[i]
#     an!(i) = J*phi[i]/2
#     b!(i)  = 0.0
#     Discretisation(phi, [Divergence{Linear}], [1], ap!, an!, b!)
# end