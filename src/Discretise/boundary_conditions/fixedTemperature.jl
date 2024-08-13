@inline (bc::FixedTemperature)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # extract user provided information
    (; T, energy_model) = bc.value

    # h = energy_model.update_BC(energy_model, T)
    h = energy_model.update_BC(T)

    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment b array     
    # Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    # nothing
    0.0, -ap*h
end

# fixedTempterature boundary condition
@inline (bc::FixedTemperature)(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # extract user provided information
    (; T, energy_model) = bc.value

    # h = energy_model.update_BC(energy_model, T)
    h = energy_model.update_BC(T)

    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])
    vol = 1#cell.volume
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment b array     
    # Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    # nothing
    -term.flux[fID], -ap*h
end

# fixedTempterature boundary condition
@inline (bc::FixedTemperature)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing
    ) where {F,P,I} = begin
    # Retrieve term flux and extract fields from workitem face
    J = term.flux[fID]
    (; area, delta) = face 

    # extract user provided information
    (; T, energy_model) = bc.value

    # h = energy_model.update_BC(energy_model, T)
    h = energy_model.update_BC(T)

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*bc.value
    # nothing
    ap, ap*h
end

@inline (bc::FixedTemperature)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing
    ) where {F,P,I} = begin
    # Retrieve term flux and extract fields from workitem face

    # extract user provided information
    (; T, energy_model) = bc.value

    # h = energy_model.update_BC(energy_model, T)
    h = energy_model.update_BC(T)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment b array     
    # Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    # nothing
    0.0, term.sign[1]*(-term.flux[fID]*h)
end