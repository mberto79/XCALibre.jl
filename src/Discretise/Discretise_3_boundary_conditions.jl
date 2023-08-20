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
    term::Operator{F,P,I,Laplacian{Linear}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} where T = begin
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
    term::Operator{F,P,I,Divergence{Linear}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{T}}, 
    A, b, cellID, cell, face, fID) where {F,P,I} where T = begin
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

@inline (bc::KWallFunction)(# a bit hacky for now (should be Src)
    term::Operator{F,P,I,Si}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    # cmu, κ, k = bc.value
    # b[cellID] -= k[cellID]^1.5*cmu^0.75/(κ*face.delta)*cell.volume
    nothing
end

@inline (bc::OmegaWallFunction)(# a bit hacky for now (should be Src)
    term::Operator{F,P,I,Si}, 
    A, b, cellID, cell, face, fID) where {F,P,I} = begin
    # cmu, κ, k = bc.value
    # ωc = k[cellID]^0.5/(cmu^0.25*κ*face.delta)*cell.volume
    # y = face.delta
    # ωc = 6*1e-3/(0.075*y^2)
    # b[cellID] += A[cellID,cellID]*ωc
    # A[cellID,cellID] += A[cellID,cellID]
    # b[cellID] += A[cellID,cellID]*ωc
    # A[cellID,cellID] += A[cellID,cellID]
    nothing
end