export dirichlet, neumann

# LAPLACIAN TERM (NON-UNIFORM)

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    A, b, cellID, cell, face, fID
    ) where {F,P,I} = begin
    J = term.flux[fID]
    (; area, delta) = face 
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    A[cellID,cellID] += ap
    b[cellID] += ap*bc.value
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    phi = term.phi 
    values = phi.values
    A[cellID,cellID] += 0.0
    b[cellID] += 0.0
    nothing
end

# DIVERGENCE TERM (NON-UNIFORM)

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    A[cellID,cellID] += 0.0 
    b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])
    A[cellID,cellID] += ap
    nothing
end


# IMPLICIT SOURCE

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Si}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Si}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    nothing
end