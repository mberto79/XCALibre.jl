export PropertyGrid, PropertyTable, table_lookup
export TabulatedEos, TabulatedMu, TabulatedK, TabulatedCp, TabulatedBeta
export SaturationCurve
export specific_gas_constant, latent_heat
export saturation_range, check_saturation_range

# =============================================================================
#  Regular (p, T) grid + bilinear lookup
# =============================================================================

"""
    PropertyGrid(; p_min, p_max, np, T_min, T_max, nT)

Uniform rectangular grid in absolute pressure [Pa] and temperature [K] used by
every [`PropertyTable`](@ref).

Stored internally as origin + spacing rather than as coordinate vectors, so the
whole struct is `isbits` and a lookup needs no memory beyond the value array
itself.

Construction is **keyword-only** on purpose. The stored layout
`(p_min, dp, np, T_min, dT, nT)` has the same arity and argument types as the
natural `(p_min, p_max, np, T_min, T_max, nT)` spelling, so a positional outer
constructor is shadowed by the compiler-generated one and `p_max` is silently
taken as the spacing — a mistake that produces a plausible-looking table
covering entirely the wrong range.
"""
struct PropertyGrid{F<:AbstractFloat, I<:Integer}
    p_min::F
    dp::F
    np::I
    T_min::F
    dT::F
    nT::I
end
Adapt.@adapt_structure PropertyGrid

function PropertyGrid(; p_min, p_max, np::Integer, T_min, T_max, nT::Integer)
    np >= 2 || throw(ArgumentError("`np` must be at least 2, got $np"))
    nT >= 2 || throw(ArgumentError("`nT` must be at least 2, got $nT"))
    p_max > p_min || throw(ArgumentError("Need p_max > p_min, got ($p_min, $p_max)"))
    T_max > T_min || throw(ArgumentError("Need T_max > T_min, got ($T_min, $T_max)"))
    F = promote_type(typeof(float(p_min)), typeof(float(T_min)))
    return PropertyGrid{F,typeof(np)}(
        F(p_min), F((p_max - p_min)/(np - 1)), np,
        F(T_min), F((T_max - T_min)/(nT - 1)), nT)
end

"""Grid node pressures/temperatures, as ranges. Host-side helpers for table building."""
grid_pressures(g::PropertyGrid) = range(g.p_min, step=g.dp, length=g.np)
grid_temperatures(g::PropertyGrid) = range(g.T_min, step=g.dT, length=g.nT)

grid_p_max(g::PropertyGrid) = g.p_min + g.dp*(g.np - 1)
grid_T_max(g::PropertyGrid) = g.T_min + g.dT*(g.nT - 1)

"""
    PropertyTable(grid, values)

A single tabulated property on a [`PropertyGrid`](@ref). `values` is `nT x np`,
i.e. temperature varies down a column, which makes the two nodes bracketing `T`
adjacent in memory — the inner axis of every lookup.

Queries outside the grid are **clamped** to the boundary value rather than
extrapolated. A Helmholtz EOS extrapolated beyond its tabulated range produces
confidently wrong numbers (and, near the critical point, non-monotonic ones), so
freezing is the safer failure mode. Build the table wide enough for the run:
`check_table_range` reports when a solution has left it.
"""
struct PropertyTable{F<:AbstractFloat, I<:Integer, A<:AbstractMatrix}
    grid::PropertyGrid{F,I}
    values::A
end
Adapt.@adapt_structure PropertyTable

"""
    table_lookup(table, p, T)

Bilinear interpolation of `table` at absolute pressure `p` and temperature `T`,
clamped to the tabulated range. Callable from a `KernelAbstractions` kernel.
"""
@inline function table_lookup(table::PropertyTable, p, T)
    (; grid, values) = table
    (; p_min, dp, np, T_min, dT, nT) = grid

    fp = (p - p_min)/dp
    fT = (T - T_min)/dT

    # Cell index in 0-based terms, held one short of the last node so that
    # `i + 1` is always a valid neighbour.
    i = unsafe_trunc(Int, clamp(floor(fp), 0, np - 2))
    j = unsafe_trunc(Int, clamp(floor(fT), 0, nT - 2))

    # Clamping the weights (rather than the inputs) is what freezes the value
    # outside the grid: off the low edge fp < 0 gives a negative weight, which
    # would otherwise extrapolate.
    wp = clamp(fp - i, zero(fp), one(fp))
    wT = clamp(fT - j, zero(fT), one(fT))

    @inbounds begin
        v00 = values[j + 1, i + 1]
        v10 = values[j + 2, i + 1]
        v01 = values[j + 1, i + 2]
        v11 = values[j + 2, i + 2]
    end

    v0 = v00 + wT*(v10 - v00)
    v1 = v01 + wT*(v11 - v01)
    return v0 + wp*(v1 - v0)
end


# =============================================================================
#  Property models backed by a table
# =============================================================================

"""
    TabulatedEos <: AbstractEosModel

Real-fluid equation of state supplied as pre-computed tables of density and
isothermal compressibility over a (p, T) grid.

This is the non-ideal alternative to [`IdealGas`](@ref) for cases where
`rho = p/(R*T)` is not defensible — notably hydrogen at a pressure that is a
substantial fraction of its critical pressure (1.2964 MPa), where the vapour
density departs from ideal by tens of percent.

Tabulation is not an optimisation but a requirement: a direct Helmholtz
evaluation root-finds for density and allocates, so it can neither run inside a
`KernelAbstractions` kernel nor be afforded per cell per time step. Build one
with `RealFluid` (see `FluidProperties/property_tables.jl`).

### Fields
- `rho`  -- Density table [kg/m^3].
- `psi`  -- Isothermal compressibility table, `(1/rho)(d rho/d p)|_T` [1/Pa].
- `R`    -- Specific gas constant [J/kg/K], carried for the kinetic prefactor of
            the `Lee`/`Schrage` phase change models. It is *not* used by the
            equation of state itself.
"""
struct TabulatedEos{F<:AbstractFloat, T1, T2} <: AbstractEosModel
    rho::T1
    psi::T2
    R::F
end
Adapt.@adapt_structure TabulatedEos

"""
    TabulatedMu <: AbstractViscosityModel

Dynamic viscosity [Pa s] as a function of (p, T).
"""
struct TabulatedMu{T} <: AbstractViscosityModel
    mu::T
end
Adapt.@adapt_structure TabulatedMu

"""
    TabulatedK <: AbstractConductivityModel

Thermal conductivity [W/m/K] as a function of (p, T).
"""
struct TabulatedK{T} <: AbstractConductivityModel
    k::T
end
Adapt.@adapt_structure TabulatedK

"""
    TabulatedCp <: AbstractHeatCapacityModel

Specific heat capacity at constant pressure [J/kg/K] as a function of (p, T).

Worth having variable rather than constant for near-critical hydrogen: between
0.4 MPa and 1.1 MPa the saturated-liquid `cp` roughly doubles, so a single
constant chosen for one operating point is badly wrong at another.
"""
struct TabulatedCp{T} <: AbstractHeatCapacityModel
    cp::T
end
Adapt.@adapt_structure TabulatedCp

"""
    TabulatedBeta <: AbstractExpansivityModel

Volumetric thermal expansivity [1/K] as a function of (p, T).
"""
struct TabulatedBeta{T} <: AbstractExpansivityModel
    beta::T
end
Adapt.@adapt_structure TabulatedBeta


# --- equation-of-state interface --------------------------------------------

phase_compressibility(eos::TabulatedEos, p_abs, T) = table_lookup(eos.psi, p_abs, T)

# `beta` is looked up per cell and passed in by the caller, so the real-fluid
# case is the plain definition `beta*T` — the same as `ConstEos`. Only
# `IdealGas` is special (beta = 1/T exactly, so the product is one).
phase_betaT(::TabulatedEos, beta, T) = beta*T

"""
    specific_gas_constant(eos) -> J/kg/K

The specific gas constant of a phase, needed by the kinetic prefactor
`sqrt(1/(2 pi R T_sat))` of the `Lee` and `Schrage` phase change models.

Defined for the equations of state that carry one; anything else returns
`nothing`, which lets `validate_phase_change_setup` produce a useful message
rather than a `MethodError`.
"""
specific_gas_constant(eos) = nothing
specific_gas_constant(eos::IdealGas) = eos.R
specific_gas_constant(eos::TabulatedEos) = eos.R


# --- per-cell property updates ----------------------------------------------

function update_phase_property!(field, model::TabulatedEos, p_abs, T, config)
    _fill_from_table!(field, model.rho, p_abs, T, config)
end

function update_phase_property!(field, model::TabulatedMu, p_abs, T, config)
    _fill_from_table!(field, model.mu, p_abs, T, config)
end

function update_phase_property!(field, model::TabulatedK, p_abs, T, config)
    _fill_from_table!(field, model.k, p_abs, T, config)
end

function update_phase_property!(field, model::TabulatedCp, p_abs, T, config)
    _fill_from_table!(field, model.cp, p_abs, T, config)
end

function update_phase_property!(field, model::TabulatedBeta, p_abs, T, config)
    _fill_from_table!(field, model.beta, p_abs, T, config)
end

function _fill_from_table!(field, table, p_abs, T, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(field)
    kernel! = _table_fill!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, table, p_abs, T)
    return nothing
end

@kernel inbounds=true function _table_fill!(field, table, p_abs, T)
    i = @index(Global)
    field[i] = table_lookup(table, p_abs[i], T[i])
end

# Every tabulated property varies per cell, so all of them need storage.
_phase_property_field(::TabulatedEos, mesh) = ScalarField(mesh)
_phase_property_field(::TabulatedMu, mesh) = ScalarField(mesh)
_phase_property_field(::TabulatedK, mesh) = ScalarField(mesh)
_phase_property_field(::TabulatedCp, mesh) = ScalarField(mesh)
_phase_property_field(::TabulatedBeta, mesh) = ScalarField(mesh)


# =============================================================================
#  Saturation curve from tabulated data
# =============================================================================

"""
    SaturationCurve <: AbstractSaturationModel

Saturation relation held as 1-D tables rather than as a correlation, so it can
come straight from the same equation of state as the bulk properties.

Two independent tables are stored because both directions are needed and
inverting one numerically inside a kernel is not an option:

- `T_sat` on a pressure grid  -- used by every phase change model,
- `p_sat` on a temperature grid -- used by `Schrage`,
- `h_fg` on a pressure grid   -- latent heat, which for hydrogen falls by more
  than half between 0.4 MPa and the critical point, so treating it as constant
  over a pressure sweep is not tenable.

Contrast with [`Antoine`](@ref), which is a closed-form fit valid over a stated
temperature window.

### Fields
- `p_min`, `dp`, `np` -- pressure grid for `T_sat` and `h_fg`.
- `T_min`, `dT`, `nT` -- temperature grid for `p_sat`.
- `T_sat`, `h_fg`, `p_sat` -- the tabulated values.
"""
struct SaturationCurve{F<:AbstractFloat, I<:Integer, V} <: AbstractSaturationModel
    p_min::F
    dp::F
    np::I
    T_min::F
    dT::F
    nT::I
    T_sat::V
    h_fg::V
    p_sat::V
end
Adapt.@adapt_structure SaturationCurve

@inline function _lookup_1d(values, x, x_min, dx, n)
    f = (x - x_min)/dx
    i = unsafe_trunc(Int, clamp(floor(f), 0, n - 2))
    w = clamp(f - i, zero(f), one(f))
    @inbounds v0 = values[i + 1]
    @inbounds v1 = values[i + 2]
    return v0 + w*(v1 - v0)
end

@inline saturation_temperature(sat::SaturationCurve, p) =
    _lookup_1d(sat.T_sat, p, sat.p_min, sat.dp, sat.np)

"""
    saturation_range(sat) -> (p_min, p_max)

The absolute-pressure interval a `SaturationCurve` was tabulated over.

`saturation_temperature` CLAMPS outside this range - it is a kernel function and
cannot throw - so a query past the edge silently returns the edge temperature.
That is how a pressure excursion becomes a spurious wall superheat: at 0.7 MPa
operating with a table starting at 0.25 MPa, a cell whose `p_abs` dips below the
bound gets `T_sat = 23.86 K` instead of 29.15 K, and the liquid then appears 5 K
superheated. `N_a ~ dT_sup^1.805` turns that into an evaporation rate far above
what the wall can supply.

Use with [`check_saturation_range`](@ref) to catch it where it happens.
"""
saturation_range(sat::SaturationCurve) = (sat.p_min, sat.p_min + sat.dp*(sat.np - 1))
saturation_range(sat) = nothing

"""
    check_saturation_range(sat, p_abs) -> nothing

Throw if any cell's absolute pressure lies outside the saturation curve's
tabulated range.

**Why this is an error and not a warning.** The clamp is silent and its effect is
large and non-obvious: it does not degrade the answer gracefully, it manufactures
several kelvin of superheat and drives a boiling model that is exponential in
superheat. A run that trips this is not slightly inaccurate, it is reporting
physics that never happened - so it should stop, the same way
`validate_property_table` stops a discontinuous density table at build time.

Cheap enough to call every step: one min/max reduction against a solve.
"""
check_saturation_range(::Any, ::Nothing) = nothing
check_saturation_range(sat, p_abs) = _check_saturation_range(saturation_range(sat), p_abs)
_check_saturation_range(::Nothing, p_abs) = nothing

function _check_saturation_range(range::Tuple, p_abs)
    p_lo, p_hi = range
    lo, hi = extrema(p_abs.values)
    (lo >= p_lo && hi <= p_hi) && return nothing
    throw(ArgumentError(
        """Absolute pressure has left the saturation curve's tabulated range.

  p_abs in the domain : $(round(lo/1e6, digits=6)) - $(round(hi/1e6, digits=6)) MPa
  saturation curve    : $(round(p_lo/1e6, digits=6)) - $(round(p_hi/1e6, digits=6)) MPa

`saturation_temperature` CLAMPS outside this range rather than failing, so the
run would continue with a wrong T_sat - at the lower bound that means a liquid at
saturation appearing superheated by several kelvin, which an RPI site density
(~dT_sup^1.805) amplifies into an evaporation rate the wall cannot supply.

Either widen the curve, e.g.

    build_saturation_curve(H2(), p=(p_lo_new, p_hi_new), T=(...), ...)

or find why the pressure is excursing - a genuine 0.7 MPa case should not visit
$(round(lo/1e6, digits=4)) MPa."""))
end

@inline saturation_pressure(sat::SaturationCurve, T) =
    _lookup_1d(sat.p_sat, T, sat.T_min, sat.dT, sat.nT)

"""
    latent_heat(sat, p, L_ref) -> J/kg

Latent heat of vaporisation at pressure `p`.

`L_ref` is the fluid's scalar `h_fg`, returned unchanged by correlations that
carry no latent heat of their own (`Antoine`). It is deliberately the fallback
rather than an error: the bulk phase change source and the energy equation's
latent-heat sink share a single scalar `L`, and that shared value is what keeps
the two consistent.
"""
@inline latent_heat(::AbstractSaturationModel, p, L_ref) = L_ref
@inline latent_heat(sat::SaturationCurve, p, L_ref) =
    _lookup_1d(sat.h_fg, p, sat.p_min, sat.dp, sat.np)
