export scheme!, scheme_source!, scheme_face_source!, is_second_order

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b" 
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

# TIME 

# SteadyState
@inline function scheme!(
    term::Operator{F,P,I,<:Time{SteadyState}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    # nothing
    z = zero(eltype(nzval_array))
    z, z
end
@inline scheme_source!(
    term::Operator{F,P,I,<:Time{SteadyState}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P,I} = begin
    z = zero(eltype(cells.volume))
    z, z
end

## Euler
@inline function scheme!(
    term::Operator{F,P,I,<:Time{Euler}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    0.0, 0.0 # add types if this approach works
end

@inline scheme_source!(
    term::Operator{F,P,I,<:Time{Euler}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:ScalarField,I} = begin
        volume = cells.volume[cID]
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]
        
        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt
        return ac, b
end
@inline scheme_source!(
    term::Operator{F,P,I,<:Time{Euler}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:VectorField,I} = begin # Special case for U_eqn (rho)
        volume = cells.volume[cID]
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]
        
        # Increment sparse and b arrays 
        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt
        return ac, b
end

## Crank-Nicolson: the matrix carries (1 + c)ρV/Δt and the source V·S, with S = (1 + c)ρxⁿ/Δt +
## c ddtⁿ set by `time_level!` (c = ψ, or 0 on the backward-Euler first step); in a kernel the
## operator's state is the 1-element array holding 1 + c
@inline function scheme!(
    term::Operator{F,P,I,<:Time{CrankNicolson}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    z = zero(eltype(nzval_array))
    z, z
end
@inline scheme_source!(
    term::Operator{F,P,I,<:Time{CrankNicolson}}, cells, cID, cIndex, prev, runtime, rho_prev
    )  where {F,P,I} = begin
        volume = cells.volume[cID]
        ac = term.type.state[1]*term.flux[cID]*volume/runtime.dt[1]
        b = prev[cID]*volume
        return ac, b
end

# LAPLACIAN

@inline function scheme!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}

    (; face_gDiff) = term.phi.mesh
    ap = term.sign*term.flux[fID]*face_gDiff[fID]

    # Increment sparse array
    ac = -ap
    an = ap
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Laplacian{Linear}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P,I} = begin
    0.0, 0.0
end

# DIVERGENCE

# Linear
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}

    w = faces.weight[fID]
    # signbit(ns) ? w = one(w) - w : w
    half = typeof(w)(0.5)
    w = half + ns*(w - half)
    
    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    ac = ap*w
    an = ap*(one(w) - w)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Linear}}, cells, cID, cIndex, prev, runtime, rho_prev) where {F,P,I} = begin
    0.0, 0.0
end

# Upwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    z = zero(ap)
    ac = max(ap, z)
    an = -max(-ap, z)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Upwind}}, cells, cID, cIndex, prev, runtime, rho_prev) where {F,P,I} = begin
    0.0, 0.0
end

# LUST
@inline function scheme!(
    term::Operator{F,P,I,Divergence{LUST}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    
    w = faces.weight[fID]
    signbit(ns) ? w = one(w) - w : w

    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    acLinear = ap*w 
    anLinear = ap*(one(w) - w)
    z = zero(ap)
    acUpwind = max(ap, z)
    anUpwind = -max(-ap, z)
    three_quarters = typeof(ap)(0.75)
    quarter = typeof(ap)(0.25)
    ac = three_quarters*acLinear + quarter*acUpwind
    an = three_quarters*anLinear + quarter*anUpwind
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{LUST}}, cells, cID, cIndex, prev, runtime, rho_prev) where {F,P,I} = begin
    0.0, 0.0
end

# LUSTDC: upwind matrix, 3/4 (linear - upwind) face value as an explicit source (see LUSTDC)
@inline scheme!(
    term::Operator{F,P,I,Divergence{LUSTDC}},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    ) where {F,P,I} = scheme!(
        _upwind(term), nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)

@inline scheme_source!(
    term::Operator{F,P,I,Divergence{LUSTDC}}, cells, cID, cIndex, prev, runtime, rho_prev
    ) where {F,P,I} = begin
    0.0, 0.0
end

@inline function scheme_face_source!(
    term::Operator{F,P,I,Divergence{LUSTDC}}, cells, faces, cID, nID, ns, fID
    ) where {F,P,I}
    flux = term.sign*term.flux[fID]*ns
    w = faces.weight[fID]
    signbit(ns) ? w = one(w) - w : w
    phiP, phiN = term.phi[cID], term.phi[nID]
    phiU = flux > zero(flux) ? phiP : phiN
    return -flux*typeof(w)(0.75)*(w*phiP + (one(w) - w)*phiN - phiU)
end

# LinearUpwind, LimitedLinearUpwind: upwind matrix, face reconstruction as an explicit source
const SecondOrderUpwind = Union{LinearUpwind,LimitedLinearUpwind}

# true for schemes whose face values need the gradient of the transported field
is_second_order(::Type) = false
is_second_order(::Type{<:SecondOrderUpwind}) = true
is_second_order(::Type{LUSTLU}) = true

@inline _upwind(term::Operator) = Operator(term.flux, term.phi, term.sign, Divergence{Upwind}())

Divergence{S}(flux, phi, grad) where S<:SecondOrderUpwind = Operator(
    flux, phi, 1, Divergence{S}(), grad)

Divergence{S}(flux, phi) where S<:SecondOrderUpwind = begin
    @warn "Divergence{$S} built without a gradient: this term reduces to Upwind" maxlog=1
    Operator(flux, phi, 1, Divergence{S}())
end

@inline scheme!(
    term::Operator{F,P,I,Divergence{S}},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    ) where {F,P,I,S<:SecondOrderUpwind} = scheme!(
        _upwind(term), nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)

@inline scheme_source!(
    term::Operator{F,P,I,Divergence{S}}, cells, cID, cIndex, prev, runtime, rho_prev
    ) where {F,P,I,S<:SecondOrderUpwind} = begin
    0.0, 0.0
end

# LUSTLU: LUST matrix, 1/4 of the LinearUpwind face reconstruction as an explicit source (see LUSTLU)
@inline _lust(term::Operator) = Operator(term.flux, term.phi, term.sign, Divergence{LUST}())

Divergence{LUSTLU}(flux, phi, grad) = Operator(flux, phi, 1, Divergence{LUSTLU}(), grad)

Divergence{LUSTLU}(flux, phi) = begin
    @warn "Divergence{LUSTLU} built without a gradient: this term reduces to LUST" maxlog=1
    Operator(flux, phi, 1, Divergence{LUSTLU}())
end

@inline scheme!(
    term::Operator{F,P,I,Divergence{LUSTLU}},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    ) where {F,P,I} = scheme!(
        _lust(term), nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)

@inline scheme_source!(
    term::Operator{F,P,I,Divergence{LUSTLU}}, cells, cID, cIndex, prev, runtime, rho_prev
    ) where {F,P,I} = scheme_source!(_lust(term), cells, cID, cIndex, prev, runtime, rho_prev)

@inline function scheme_face_source!(
    term::Operator{F,P,I,Divergence{LUSTLU}}, cells, faces, cID, nID, ns, fID
    ) where {F,P,I}
    flux = term.sign*term.flux[fID]*ns
    uID = flux > zero(flux) ? cID : nID
    d = faces.centre[fID] - cells.centre[uID]
    return -flux*typeof(flux)(0.25)*_reconstruct(term.grad[uID], d)
end

# explicit contribution of an internal face (fID) of cell cID to the right-hand side; only operators
# for which `has_face_source` holds are visited, all others compile to the original code
has_face_source(::Type{<:Operator}) = false
has_face_source(::Type{<:Operator{F,P,I,Divergence{S},G}}) where {F,P,I,S<:SecondOrderUpwind,G<:AbstractField} = true
has_face_source(::Type{<:Operator{F,P,I,Divergence{LUSTDC}}}) where {F,P,I} = true
has_face_source(::Type{<:Operator{F,P,I,Divergence{LUSTLU},G}}) where {F,P,I,G<:AbstractField} = true

@inline function scheme_face_source!(
    term::Operator{F,P,I,Divergence{LinearUpwind}}, cells, faces, cID, nID, ns, fID
    ) where {F,P,I}
    flux = term.sign*term.flux[fID]*ns
    uID = flux > zero(flux) ? cID : nID
    d = faces.centre[fID] - cells.centre[uID]
    return -flux*_reconstruct(term.grad[uID], d)
end

@inline function scheme_face_source!(
    term::Operator{F,P,I,Divergence{LimitedLinearUpwind}}, cells, faces, cID, nID, ns, fID
    ) where {F,P,I}
    flux = term.sign*term.flux[fID]*ns
    uID, dID = flux > zero(flux) ? (cID, nID) : (nID, cID)
    xU = cells.centre[uID]
    grad = term.grad[uID]
    δ = _reconstruct(grad, faces.centre[fID] - xU)
    g = _reconstruct(grad, cells.centre[dID] - xU)
    Δ = term.phi[dID] - term.phi[uID]
    return -flux*_limit_tvd.(δ, g, Δ)
end

@inline _reconstruct(grad::SVector, d) = grad⋅d
@inline _reconstruct(grad::SMatrix, d) = grad*d

# keep the correction within the TVD region, 0 ≤ δ/Δ ≤ min(r, 1), r = 2g/Δ - 1, so that the
# face value lies between the upwind and downwind values and local extrema fall back to upwind
@inline function _limit_tvd(δ, g, Δ)
    z = zero(δ)
    iszero(Δ) && return z
    r = 2*g/Δ - one(δ)
    r > z ? Δ*clamp(δ/Δ, z, min(r, one(δ))) : z
end

# BoundedUpwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # $$\mathcal{D}_{bounded} = \sum_f \phi_f \psi_f - \psi_P \sum_f \phi_f$$
    # phif =  max(phif, 0) - max(-phi_f, 0)$
    # phif psif =  max(phif, 0) psi_P - max(-phi_f, 0)$ psi_N
    ap = term.sign*(term.flux[fID]*ns)
    z = zero(ap)
    ac = max(-ap, z)
    an = -max(-ap, z)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, cells, cID, cIndex, prev, runtime, rho_prev) where {F,P,I} = begin
    0.0, 0.0
end

# Face value of a divergence scheme for an explicitly convected field: flux*value is the face flux the
# scheme's coefficients give (w: owner weight). BoundedUpwind interpolates as Upwind; its extra
# -phi_P*sum(flux) part is a diagonal term, not a face value.
@inline convected_face_value(::Divergence{Linear}, phiP, phiN, w, flux) = w*phiP + (one(w) - w)*phiN

@inline convected_face_value(::Divergence{<:Union{Upwind,BoundedUpwind}}, phiP, phiN, w, flux) =
    flux >= zero(flux) ? phiP : phiN

@inline function convected_face_value(::Divergence{<:Union{LUST,LUSTDC}}, phiP, phiN, w, flux)
    phiUpwind = flux >= zero(flux) ? phiP : phiN
    three_quarters = typeof(w)(0.75)
    quarter = typeof(w)(0.25)
    three_quarters*(w*phiP + (one(w) - w)*phiN) + quarter*phiUpwind
end

# IMPLICIT SOURCE
@inline function scheme!(
    term::Operator{F,P,I,Si}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    z = zero(eltype(nzval_array))
    z, z
end
@inline scheme_source!(
    term::Operator{F,P,I,Si}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P,I} = begin
    
    # Retrieve and calculate flux for cell 
    flux = term.sign*term.flux[cID]*cells.volume[cID] # indexed with cID
    ac = flux # indexed with cIndex
    ac, zero(ac)
end
