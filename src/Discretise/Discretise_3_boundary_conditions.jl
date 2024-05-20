export dirichlet, neumann

# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing
    ) where {F,P,I,T} = begin
    nothing
end

# LAPLACIAN TERM (NON-UNIFORM)

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing
    ) where {F,P,I} = begin
    # Retrieve term flux and extract fields from workitem face
    J = term.flux[fID]
    (; area, delta) = face 

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    Atomix.@atomic nzval[nIndex] += ap
    Atomix.@atomic b[cellID] += ap*bc.value
    nothing
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
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
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse and b arrays
    Atomix.@atomic nzval[nIndex] += ap
    Atomix.@atomic b[cellID] += ap*values[cellID]
    nothing
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I,T}  = begin
    nothing
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I,T} = begin
    nothing
end

# DIVERGENCE TERM (NON-UNIFORM)

# Linear

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Increment b array     
    Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    nothing
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # use upwind for all
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])
    
    # Set index for sparse array values at [cellID, cellID] for workitem
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # might need to change this!!!!
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I}  = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

# Upwind

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Increment b array
    Atomix.@atomic b[cellID] -= ap*bc.value
    nothing
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Retrieve  term field and calculate ap value to increment
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    Atomix.@atomic nzval[nIndex] += ap
    # Atomix.@atomic b[cellID] += max(-ap*phi[cellID], 0.0)
    nothing
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}},
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array if ap value is positive
    Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    nothing
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, # might need to change this!!!!
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I}  = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [CellID, CellID] for workitem
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Increment sparse array if ap value is positive
    Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    nothing
end

# IMPLICIT SOURCE

# Dirichlet functor definition
@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    nothing
end

# Neumann functor definition
@inline (bc::Neumann)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    nothing
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    nothing
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione, component=nothing) where {F,P,I} = begin
    # Retrieve workitem term field
    phi = term.phi[cellID] 

    # Calculate flux to increment
    flux = term.sign*term.flux[cellID]

    # Incrememnt b array
    Atomix.@atomic b[cellID] += flux*phi*cell.volume 
    nothing
end