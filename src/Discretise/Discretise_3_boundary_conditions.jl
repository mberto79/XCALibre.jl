export dirichlet, neumann

# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, cellID, zcellID, cell, face, fID, ione, component=nothing
    ) where {F,P,I,T} = begin
    # nothing
    0.0, 0.0 # need to add consistent return types
end

# LAPLACIAN TERM (NON-UNIFORM)

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing
    ) where {F,P,I} = begin
    # Retrieve term flux and extract fields from workitem face
    J = term.flux[fID]
    (; area, delta) = face 

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*bc.value
    # nothing
    ap, ap*bc.value
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
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
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    # nothing
    ap, ap*values[cellID]
end

# Wall functor definition
@inline (bc::Wall)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
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
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    # nothing
    ap, ap*(U_boundary[component.value] + norm_vel[component.value])
end

# Symmetry functor definition
@inline (bc::Symmetry)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Retrive term field and field values

    phi = term.phi 

    velocity_cell = phi[cellID]
    J = term.flux[fID]

    # Extract required fields from workitem face
    (; area, delta, normal) = face 

    # Calculate wall normal velocity at cell centre
    norm_vel_ex_comp = ((velocity_cell⋅normal))*normal[component.value]
    
    # norm_vel_ex_comp = norm_vel_ex[component.value]

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    # nothing
    0.0, 0.5*ap*(norm_vel_ex_comp)
    # -ap, ap*(norm_vel_ex_comp-velocity_cell[component.value])
    # (1.0)*ap*normal[component.value]*normal[component.value], ap*(norm_vel_ex_comp-velocity_cell[component.value]*normal[component.value]*normal[component.value])
end

# fixedTempterature boundary condition
@inline (bc::FixedTemperature)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing
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
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*bc.value
    # nothing
    ap, ap*h
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I,T}  = begin
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
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    # nothing
    ap, ap*values[cellID]
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I,T} = begin
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
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    # Atomix.@atomic nzval[zcellID] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    # nothing
    ap, ap*values[cellID]
end

# DIVERGENCE TERM (NON-UNIFORM)

# Linear

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Increment b array     
    # Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    # nothing
    0.0, term.sign[1]*(-term.flux[fID]*bc.value)
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    ap, 0.0
end

# Neumann functor definition
@inline (bc::Wall)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    0.0, 0.0
end

@inline (bc::Symmetry)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    0.0, 0.0
end


# fixedTempterature boundary condition
@inline (bc::FixedTemperature)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing
    ) where {F,P,I} = begin
    # Retrieve term flux and extract fields from workitem face

    # extract user provided information
    (; T, energy_model) = bc.value

    # h = energy_model.update_BC(energy_model, T)
    h = energy_model.update_BC(T)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment b array     
    # Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    # nothing
    0.0, term.sign[1]*(-term.flux[fID]*h)
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    ap, 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I}  = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    ap, 0.0
end

# Upwind

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Increment b array
    # Atomix.@atomic b[cellID] -= ap*bc.value
    # nothing
    0.0, -ap*bc.value
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Retrieve  term field and calculate ap value to increment
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    ap, 0.0
end

# Neumann functor definition
@inline (bc::Wall)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    0.0, 0.0
end

# Neumann functor definition
@inline (bc::Symmetry)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    0.0, 0.0
end

# fixedTempterature boundary condition
@inline (bc::FixedTemperature)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # extract user provided information
    (; T, energy_model) = bc.value

    # h = energy_model.update_BC(energy_model, T)
    h = energy_model.update_BC(T)

    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment b array     
    # Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    # nothing
    0.0, -ap*h
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Retrieve  term field and calculate ap value to increment
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    # ap, 0.0
    0.0, 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I}  = begin
    # Retrieve  term field and calculate ap value to increment
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    # ap, 0.0
    0.0, 0.0
end

# IMPLICIT SOURCE

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Retrieve workitem term field
    phi = term.phi[cellID] 

    # Calculate flux to increment
    flux = term.sign*term.flux[cellID]

    # Incrememnt b array
    # Atomix.@atomic b[cellID] += flux*phi*cell.volume 
    # nothing
    # 0.0, flux*phi*cell.volume
    0.0, 0.0
end


