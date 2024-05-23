export dirichlet, neumann

# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing
    ) where {F,P,I,T} = begin
    # nothing
    0.0, 0.0 # need to add consistent return types
end

# LAPLACIAN TERM (NON-UNIFORM)

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing
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
    term::Operator{F,P,I,Laplacian{Linear}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
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

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I,T}  = begin
    # nothing
    0.0, 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I,T} = begin
    # nothing
    0.0, 0.0
end

# DIVERGENCE TERM (NON-UNIFORM)

# Linear

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Increment b array     
    # Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    # nothing
    0.0, term.sign[1]*(-term.flux[fID]*bc.value)
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    # Atomix.@atomic nzval[zcellID] += ap
    # nothing
    ap, 0.0
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # use upwind for all
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
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
    term::Operator{F,P,I,Divergence{Linear}}, # might need to change this!!!!
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I}  = begin
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
    term::Operator{F,P,I,Divergence{Upwind}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Increment b array
    # Atomix.@atomic b[cellID] -= ap*bc.value
    # nothing
    0.0, -ap*bc.value
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
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

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}},
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array if ap value is positive
    # Atomix.@atomic nzval[cellID, zcellID] += max(ap, 0.0)
    # nothing
    max(ap, 0.0), 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, # might need to change this!!!!
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I}  = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array if ap value is positive
    # Atomix.@atomic nzval[zcellID] += max(ap, 0.0)
    # nothing
    max(ap, 0.0), 0.0
end

# IMPLICIT SOURCE

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Retrieve workitem term field
    phi = term.phi[cellID] 

    # Calculate flux to increment
    flux = term.sign*term.flux[cellID]

    # Incrememnt b array
    # Atomix.@atomic b[cellID] += flux*phi*cell.volume 
    # nothing
    0.0, flux*phi*cell.volume
end