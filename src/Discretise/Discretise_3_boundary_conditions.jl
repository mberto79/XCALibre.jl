
# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,T,TF} = begin
    # nothing
    z = zero(TF)
    z, z
end

# second-order upwind: boundary faces take upwind values, so every condition reuses its Upwind method
@inline (bc::AbstractBoundary)(
    term::Operator{F,P,I,Divergence{S}}, colval, rowptr, nzval, cellID, zcellID,
    cells, faces, fID, i, component, time
    ) where {F,P,I,S<:SecondOrderUpwind} = bc(
        _upwind(term), colval, rowptr, nzval, cellID, zcellID, cells, faces, fID, i, component, time)

# LUSTDC: boundary faces take the LUST coefficients, so the converged state is that of LUST
@inline (bc::AbstractBoundary)(
    term::Operator{F,P,I,Divergence{LUSTDC}}, colval, rowptr, nzval, cellID, zcellID,
    cells, faces, fID, i, component, time
    ) where {F,P,I} = bc(
        Operator(term.flux, term.phi, term.sign, Divergence{LUST}()),
        colval, rowptr, nzval, cellID, zcellID, cells, faces, fID, i, component, time)

# KWallFunction
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,T,TF}  = begin
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    area, delta = faces.area[fID], faces.delta[fID]
    flux = -J*area/delta
    ap = term.sign*(flux)
    ap, ap*values[cellID] # original
    z = zero(TF)
    z, z
end

# OmegaWallFunction
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,T,TF} = begin
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    area, delta = faces.area[fID], faces.delta[fID]
    flux = -J*area/delta
    ap = term.sign*(flux)
    TF(ap), TF(ap*values[cellID]) # original
    # 0.0, 0.0
end

# KWallFunction 
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF} = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    TF(ap), zero(TF) # original
end

# OmegaWallFunction 
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF}  = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    TF(ap), zero(TF) # original
end

# KWallFunction
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{LUST}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF} = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    TF(ap), zero(TF) # original
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF} = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    TF(ap), zero(TF) # original
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF} = begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    TF(ap-flux), zero(TF)
end

# OmegaWallFunction
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{LUST}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF}  = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    TF(ap), zero(TF) # original
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF}  = begin
    flux = term.flux[fID]
    ap = term.sign*(flux) 
    TF(ap), zero(TF) # original
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF}  = begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    TF(ap-flux), zero(TF)
end

# KWallFunction 
@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF} = begin
    z = zero(TF)
    z, z
end

# OmegaWallFunction 
@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF} = begin

    # phi = term.phi[cellID] 
    # flux = term.sign*term.flux[cellID]

    z = zero(TF)
    z, z
end
