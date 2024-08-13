# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Retrive term field and field values
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]

    # Extract required fields from workitem face
    (; area, delta) = face 

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    # nothing
    ap, ap*values[cellID]
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    ap, 0.0
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Retrieve  term field and calculate ap value to increment
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)
    # Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    ap, 0.0
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Retrieve  term field and calculate ap value to increment
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])
    vol = 1#cell.volume

    # Set index for sparse array values at [CellID, CellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)
    # Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    ap-term.flux[fID], 0.0
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end