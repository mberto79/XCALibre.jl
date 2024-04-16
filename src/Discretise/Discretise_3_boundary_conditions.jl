export dirichlet, neumann

# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione
    ) where {F,P,I,T} = begin
    nothing
end

# LAPLACIAN TERM (NON-UNIFORM)

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione
    ) where {F,P,I} = begin
    # Previous
    # J = term.flux[fID]
    # (; area, delta) = face 
    # flux = J*area/delta
    # ap = term.sign[1]*(-flux)
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += ap
    # Atomix.@atomic b[cellID] += ap*bc.value

    # test
    J = term.flux[fID]
    (; area, delta) = face 
    flux = J*area/(2*delta)
    ap = term.sign[1]*(-flux)
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += ap
    Atomix.@atomic b[cellID] += ap*bc.value
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    phi = term.phi 
    # # values = phi.values
    # fzero = zero(eltype(b))
    # A[cellID,cellID] += fzero
    # b[cellID] += fzero

    # previous (semi-implicit)
    # values = phi.values
    # J = term.flux[fID]
    # (; area, delta) = face 
    # flux = J*area/delta
    # ap = term.sign[1]*(-flux)
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]

    # Test
    # values = phi.values
    # J = term.flux[fID]
    # (; area, delta) = face 
    # flux = J*area/delta
    # ap = term.sign[1]*(-flux)
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += ap
    # # Atomix.@atomic b[cellID] += ap*values[cellID]
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I,T}  = begin
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I,T} = begin
    nothing # should this be Dirichlet?
end

# DIVERGENCE TERM (NON-UNIFORM)

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    # A[cellID,cellID] += 0.0 
    Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # use upwind for all
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # might need to change this!!!!
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I}  = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    # A[cellID,cellID] += 0.0 
    # b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)

    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += max(ap, 0.0)
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # nzval[nIndex] += max(ap, 0.0)
    Atomix.@atomic b[cellID] -= ap*bc.value

    # ap = term.sign[1]*(term.flux[fID])
    # b[cellID] += A[cellID,cellID]*bc.value
    # A[cellID,cellID] += A[cellID,cellID]
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    phi = term.phi 
    # ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    # ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    # b[cellID] -= ap*phi[cellID]

    # previous
    # ap = term.sign[1]*(term.flux[fID])
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    # Atomix.@atomic b[cellID] += max(-ap*phi[cellID], 0.0)

    # test
    ap = term.sign[1]*(term.flux[fID])
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    Atomix.@atomic nzval[nIndex] += ap
    # Atomix.@atomic b[cellID] += max(-ap*phi[cellID], 0.0)
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}},
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += max(ap, 0.0)
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, # might need to change this!!!!
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I}  = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += max(ap, 0.0)
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    nothing
end

# IMPLICIT SOURCE

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    # phi = term.phi[cellID] 
    # flux = term.sign*term.flux[cellID]
    # b[cellID] += flux*phi*cell.volume 
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    phi = term.phi[cellID] 
    flux = term.sign*term.flux[cellID]
    Atomix.@atomic b[cellID] += flux*phi*cell.volume 
    nothing
end

