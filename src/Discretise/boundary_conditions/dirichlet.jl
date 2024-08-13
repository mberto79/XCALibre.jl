@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing
    ) where {F,P,I} = begin
    # Retrieve term flux and extract fields from workitem face
    J = term.flux[fID]
    (; area, delta) = face 

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*bc.value
    # nothing
    ap, ap*bc.value
end

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Increment b array     
    # Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    # nothing
    0.0, term.sign[1]*(-term.flux[fID]*bc.value)
end

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])
    vol = 1#cell.volume

    # Increment b array
    # Atomix.@atomic b[cellID] -= ap*bc.value
    # nothing
    # 0.0, -ap*bc.value 
    -term.flux[fID], -ap*bc.value
end

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Increment b array
    # Atomix.@atomic b[cellID] -= ap*bc.value
    # nothing
    0.0, -ap*bc.value
end