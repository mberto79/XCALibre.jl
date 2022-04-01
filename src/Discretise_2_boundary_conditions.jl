export dirichlet, neumann

@inline dirichlet(term::Laplacian{Linear}, A, b, cellID, cell, face, value) = begin
    A[cellID,cellID] += term.sign[1]*(-term.J*face.area/face.delta)
    b[cellID] += term.sign[1]*(-term.J*face.area/face.delta*value)
    nothing
end

@inline neumann(term::Laplacian{Linear}, A, b, cellID, cell, face, value) = begin
    A[cellID,cellID] += 0.0*term.sign[1]*(-term.J*face.area/face.delta)
    b[cellID] += 0.0*term.sign[1]*(-term.J*face.area/face.delta*value)
    nothing
end

@inline dirichlet(term::Divergence{Linear}, A, b, cellID, cell, face, value) = begin
    A[cellID,cellID] += 0.0*term.sign[1]*(term.J⋅face.normal*face.area)/2.0
    b[cellID] += term.sign[1]*(-term.J⋅face.normal*face.area*value)
    nothing
end

@inline neumann(term::Divergence{Linear}, A, b, cellID, cell, face, value) = begin
    A[cellID,cellID] += term.sign[1]*(term.J⋅face.normal*face.area)
    b[cellID] += term.sign[1]*(-term.J⋅face.normal*face.area)
    nothing
end