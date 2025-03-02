
# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
    ) where {F,P,I,T} = begin
    # nothing
    0.0, 0.0 # need to add consistent return types
end

# KWallFunction
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I,T}  = begin
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
    ap, ap*values[cellID] # original
    0.0, 0.0
end

# OmegaWallFunction
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I,T} = begin
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 
    flux = -J*area/delta
    ap = term.sign*(flux)
    ap, ap*values[cellID] # original
    # 0.0, 0.0
end

# KWallFunction 
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

# OmegaWallFunction 
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I}  = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

# KWallFunction
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{LUST}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

# OmegaWallFunction
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{LUST}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I}  = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I}  = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    ap, 0.0 # original
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I}  = begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

# KWallFunction 
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin
    0.0, 0.0
end

# OmegaWallFunction 
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = begin

    # phi = term.phi[cellID] 
    # flux = term.sign*term.flux[cellID]

    0.0, 0.0
end
