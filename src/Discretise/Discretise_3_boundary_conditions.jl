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
    # values = phi.values
    A[cellID,cellID] += 0.0
    b[cellID] += 0.0
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    A, b, cellID, cell, face, fID) where {F,P,I,T}  = begin
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    A, b, cellID, cell, face, fID) where {F,P,I,T} = begin
    nothing # should this be Dirichlet?
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
    ap = term.sign[1]*(term.flux[fID])
    A[cellID,cellID] += ap
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # use upwind for all
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    ap = term.sign[1]*(term.flux[fID])
    A[cellID,cellID] += ap
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # might need to change this!!!!
    A, b, cellID, cell, face, fID) where {F,P,I,T}  = begin
    ap = term.sign[1]*(term.flux[fID])
    A[cellID,cellID] += ap
    nothing
end

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    # A[cellID,cellID] += 0.0 
    # b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)

    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += max(ap, 0.0)
    b[cellID] -= ap*bc.value

    # ap = term.sign[1]*(term.flux[fID])
    # b[cellID] += A[cellID,cellID]*bc.value
    # A[cellID,cellID] += A[cellID,cellID]
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    phi = term.phi 
    # ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    # ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    # b[cellID] -= ap*phi[cellID]
    ap = term.sign[1]*(term.flux[fID])
    A[cellID,cellID] += max(ap, 0.0)
    # b[cellID] -= max(ap*phi[cellID], 0.0)
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}},
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    ap = term.sign[1]*(term.flux[fID])
    A[cellID,cellID] += max(ap, 0.0)
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, # might need to change this!!!!
    A, b, cellID, cell, face, fID) where {F,P,I,T}  = begin
    ap = term.sign[1]*(term.flux[fID])
    A[cellID,cellID] += max(ap, 0.0)
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

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    nothing
end