export dirichlet, neumann

@inline dirichlet(term::Laplacian{Linear}, A, b, cellID, cell, face, value) = begin
    A[cellID,cellID] += term.sign[1]*(-term.J*face.area/face.delta)
    b[cellID] += term.sign[1]*(-term.J*face.area/face.delta*value)
    nothing
end

@inline neumann(term::Laplacian{Linear}, A, b, cellID, cell, face, value) = begin
    A[cellID,cellID] += term.sign[1]*(-term.J*face.area/face.delta)
    b[cellID] += term.sign[1]*(-term.J*face.area/face.delta*value)
    nothing
end