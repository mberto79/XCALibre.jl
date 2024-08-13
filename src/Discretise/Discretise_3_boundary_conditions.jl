
# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, cellID, zcellID, cell, face, fID, i, component=nothing
    ) where {F,P,I,T} = begin
    # nothing
    0.0, 0.0 # need to add consistent return types
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I,T}  = begin
    # Retrive term field and field values
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]

    # Extract required fields from workitem face
    (; area, delta) = face 

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    ap, ap*values[cellID]
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I,T} = begin
    # Retrive term field and field values
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]

    # Extract required fields from workitem face
    (; area, delta) = face 

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    ap, ap*values[cellID]
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    ap, 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I}  = begin
    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Set index for sparse array values at [cellID, cellID] for workitem
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, i)

    # Increment sparse array
    ap, 0.0
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Retrieve  term field and calculate ap value to increment
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])

    0.0, 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I}  = begin
    # Retrieve  term field and calculate ap value to increment
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])

    0.0, 0.0
end

# KWallFunction functor definition
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # nothing
    0.0, 0.0
end

# OmegaWallFunction functor definition
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin
    # Retrieve workitem term field
    phi = term.phi[cellID] 

    # Calculate flux to increment
    flux = term.sign*term.flux[cellID]

    0.0, 0.0
end
