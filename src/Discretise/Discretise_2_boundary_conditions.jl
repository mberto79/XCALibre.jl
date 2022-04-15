export dirichlet, neumann

@inline (bc::Dirichlet)(term::Laplacian{Linear}, A, b, cellID, cell, face) = begin
    ap = term.sign[1]*(-term.J*face.area/face.delta)
    A[cellID,cellID] += ap
    b[cellID] += ap*bc.value
    nothing
end

@inline (bc::Neumann)(term::Laplacian{Linear}, A, b, cellID, cell, face) = begin
    A[cellID,cellID] += 0.0*term.sign[1]*(-term.J*face.area/face.delta)
    b[cellID] += 0.0*term.sign[1]*(-term.J*face.area/face.delta*bc.value)
    nothing
end

@inline (bc::Dirichlet)(term::Divergence{Linear}, A, b, cellID, cell, face) = begin
    A[cellID,cellID] += 0.0*term.sign[1]*(term.J⋅face.normal*face.area)/2.0
    b[cellID] += term.sign[1]*(-term.J⋅face.normal*face.area*bc.value)
    nothing
end

@inline (bc::Neumann)(term::Divergence{Linear}, A, b, cellID, cell, face) = begin
    ap = term.sign[1]*(term.J⋅face.normal*face.area)
    A[cellID,cellID] += ap
    b[cellID] += -ap
    nothing
end

# @inline dirichlet(term::Laplacian{Linear}, A, b, cellID, cell, face, value) = begin
#     ap = term.sign[1]*(-term.J*face.area/face.delta)
#     A[cellID,cellID] += ap
#     b[cellID] += ap*value
#     nothing
# end

# @inline neumann(term::Laplacian{Linear}, A, b, cellID, cell, face, value) = begin
#     A[cellID,cellID] += 0.0*term.sign[1]*(-term.J*face.area/face.delta)
#     b[cellID] += 0.0*term.sign[1]*(-term.J*face.area/face.delta*value)
#     nothing
# end

# @inline dirichlet(term::Divergence{Linear}, A, b, cellID, cell, face, value) = begin
#     A[cellID,cellID] += 0.0*term.sign[1]*(term.J⋅face.normal*face.area)/2.0
#     b[cellID] += term.sign[1]*(-term.J⋅face.normal*face.area*value)
#     nothing
# end

# @inline neumann(term::Divergence{Linear}, A, b, cellID, cell, face, value) = begin
#     ap = term.sign[1]*(term.J⋅face.normal*face.area)
#     A[cellID,cellID] += ap
#     b[cellID] += -ap
#     nothing
# end