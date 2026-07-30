export FixedHeatFlux


"""
    FixedHeatFlux <: AbstractNeumann

Fixed wall heat flux boundary condition for a temperature field.

# Inputs
- `ID` Name of the boundary given as a symbol (e.g. `:tankWall`). Internally it
  gets replaced with the boundary index ID.
- `value` Heat flux in W/m^2, **positive into the domain**.

# Example
    FixedHeatFlux(:tankWall, 3.5)

# Implementation

For a temperature equation containing `- Laplacian{scheme}(keff, T)`, the
diffusive term integrated over a boundary cell contributes

    -sum_f keff * area * (dT/dn)_out

and a prescribed inward flux `q` fixes `keff * (dT/dn)_out = q` on that face
(the conductive flux vector is `-keff*grad(T)`, so heat flowing *in* means the
outward normal gradient is positive). The face contribution is therefore the
known constant `-q*area` on the left-hand side, i.e. `+q*area` on the right.

The coefficient is independent of `keff`: prescribing the flux prescribes the
whole term. That also means `q = 0` reduces exactly to `Zerogradient`, which
returns `(0, 0)` — a useful cross-check on the sign.

Contrast with `FixedTemperature`, which prescribes the wall temperature and lets
the flux follow.
"""
struct FixedHeatFlux{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure FixedHeatFlux

# The two-argument form `FixedHeatFlux(:tankWall, 3.5)` comes from the generic
# `(::Type{T})(name::Symbol, value) where T<:AbstractBoundary` constructor in
# Discretise_4_assign_boundaries.jl; `assign` then fills in `IDs_range`.


@define_boundary FixedHeatFlux Laplacian{Linear} begin
    (; area) = face
    # No dependence on the cell value, so nothing on the diagonal; the whole
    # term is a known source. `term.sign` is carried explicitly so the result
    # stays correct if the Laplacian ever appears with a `+` sign (for the usual
    # `- Laplacian` diffusion term, term.sign = -1 and this gives +q*area).
    0.0, -term.sign*bc.value*area
end

# Convection at a wall: there is no mass flux through it, so these only matter
# for robustness. Treated as zero-gradient (face value = cell value), matching
# `Zerogradient`.
@define_boundary FixedHeatFlux Divergence{Linear} begin
    ap = term.sign*(term.flux[fID])
    ap, 0.0
end

@define_boundary FixedHeatFlux Divergence{Upwind} begin
    ap = term.sign*(term.flux[fID])
    ap, 0.0
end

@define_boundary FixedHeatFlux Divergence{LUST} begin
    ap = term.sign*(term.flux[fID])
    ap, 0.0
end

# Bounded: upwind (ap, 0) minus ap on the diagonal -> (0, 0)
@define_boundary FixedHeatFlux Divergence{BoundedUpwind} begin
    0.0, 0.0
end

@define_boundary FixedHeatFlux Si begin
    0.0, 0.0
end
