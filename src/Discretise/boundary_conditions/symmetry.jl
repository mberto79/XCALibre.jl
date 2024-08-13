# Symmetry functor definition - Moukalled et al. 2016 Implementation of Boundary conditions in the finite-volume pressure-based method - Part 1
# http://dx.doi.org/10.1080/10407790.2016.1138748
@inline (bc::Symmetry)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Retrive term field and field values

    phi = term.phi 

    velocity_cell = phi[cellID]
    J = term.flux[fID]

    # Extract required fields from workitem face
    (; area, delta, normal) = face 

    # Calculate wall normal velocity at cell centre
    norm_vel= (velocity_cell⋅normal)

    # Normal velocty minus component contribution
    norm_vel = norm_vel - velocity_cell[component.value]*normal[component.value]

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    # nothing
    (2.0)*ap*normal[component.value]*normal[component.value], (2.0)*ap*(norm_vel*normal[component.value])
end

@inline (bc::Symmetry)(
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

@inline (bc::Symmetry)(
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