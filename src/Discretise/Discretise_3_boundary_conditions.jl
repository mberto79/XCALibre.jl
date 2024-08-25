
# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, cellID, zcellID, cell, face, fID, i, component, time
    ) where {F,P,I,T} = begin
    # nothing
    0.0, 0.0 # need to add consistent return types
end

# KWallFunction
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I,T}  = begin

    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 

    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    ap, ap*values[cellID]
end

# OmegaWallFunction
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I,T} = begin
    # Retrive term field and field values
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 

    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    ap, ap*values[cellID]
end

# KWallFunction 
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin

    ap = term.sign[1]*(term.flux[fID])
    ap, 0.0
end

# OmegaWallFunction 
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I}  = begin

    ap = term.sign[1]*(term.flux[fID])
    ap, 0.0
end

# KWallFunction
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])

    0.0, 0.0
end

# OmegaWallFunction
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I}  = begin

    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end

# KWallFunction 
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin

    0.0, 0.0
end

# OmegaWallFunction 
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin

    phi = term.phi[cellID] 
    flux = term.sign*term.flux[cellID]

    0.0, 0.0
end
