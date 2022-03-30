export Laplacian, Divergence

function Laplacian{Linear}(J, phi)
    ap!(cell, face, nsign, cID) = (-J * face.area)/face.delta
    an!(cell, face, nsign, cID, nID) = (J * face.area)/face.delta
    b!(cell, cID) =  0.0
    Discretisation(phi, [Laplacian{Linear}], [1], ap!, an!, b!)
end

function Divergence{Linear}(J, phi)
    ap!(i) = J*phi[i]
    an!(i) = J*phi[i]/2
    b!(i)  = 0.0
    Discretisation(phi, [Divergence{Linear}], [1], ap!, an!, b!)
end