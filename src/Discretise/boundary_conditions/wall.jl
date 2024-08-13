@inline (bc::Wall)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Retrive term field and field values

    phi = term.phi 

    # # Finding U_boundary is messy, you can index with bc.ID because they aren't in order
    # println(phi.BCs[:][1])
    
    # println(bc.ID)
    # U_boundary = phi.BCs[bc.ID].value # user given vector
    U_boundary = SVector{3}(0.0,0.0,0.0) # user given vector

    # values = get_values(phi, component)

    # println(phi.BCs[bc.ID])

    velocity_diff = phi[cellID] .- U_boundary

    J = term.flux[fID]

    # Extract required fields from workitem face
    (; area, delta, normal) = face 

    # Calculate wall normal velocity at cell centre
    norm_vel = (velocity_diff⋅normal)*normal

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    # nothing
    ap, ap*(U_boundary[component.value] + norm_vel[component.value])
end

@inline (bc::Wall)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    0.0, 0.0
end

@inline (bc::Wall)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    0.0, 0.0
end

@inline (bc::Wall)(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])
    vol = 1#cell.volume

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    -term.flux[fID], 0.0
end