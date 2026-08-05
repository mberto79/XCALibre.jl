export RealFluid, RealFluidProperties
export build_property_tables, build_saturation_curve
export table_range_report, phase_properties_at
export validate_property_table

# =============================================================================
#  Building (p, T) property tables from the Helmholtz equation of state
# =============================================================================
#
#  Everything here runs ONCE, on the host, at case setup. It root-finds and
#  allocates freely; none of it is callable from a kernel. What it produces -
#  `TabulatedEos` and friends - is what the solver actually evaluates, and that
#  is `isbits` plus one array each.
#
#  Molar-to-mass conversions are done explicitly here rather than through
#  `params_computation`, whose `conversion_factor` is applied to `beta` as well
#  (flagged "NOT TESTED!!!!" in that function) even though the isobaric
#  expansion coefficient is already 1/K and must not be rescaled. Deriving the
#  mass-specific values directly from the molar ones keeps this path independent
#  of that question.
#
#      rho [kg/m^3]   = rho_molar * M
#      cp  [J/kg/K]   = c_p(delta, tau) / M
#      h   [J/kg]     = enthalpy_calc(...) / M
#      beta[1/K]      = beta_calc(...)              (already intensive)
#      psi [1/Pa]     = k_T(T, rho_molar)           (isothermal compressibility,
#                                                    identical molar or mass)
# =============================================================================

"""
    RealFluidProperties

The five property models of one phase of a real fluid, all sharing a single
(p, T) grid. Produced by [`RealFluid`](@ref); pass to `Phase`.
"""
struct RealFluidProperties{E,V,K,C,B}
    rho::E
    mu::V
    k::K
    cp::C
    beta::B
end

"""
    Phase(rf::RealFluidProperties)

Build a `Phase` from a full set of tabulated real-fluid properties.

Equivalent to spelling out
`Phase(rho=rf.rho, mu=rf.mu, k=rf.k, cp=rf.cp, beta=rf.beta)`.
"""
Phase(rf::RealFluidProperties) =
    Phase(rho=rf.rho, mu=rf.mu, k=rf.k, cp=rf.cp, beta=rf.beta)

"""
    RealFluid(fluid, branch; p, T, np=81, nT=81, verbose=true)

Tabulate the liquid or vapour branch of a real fluid over a rectangular grid in
absolute pressure and temperature, returning the property models to hand to a
`Phase`.

### Arguments
- `fluid`  -- `H2()`, `H2_para()` or `N2()`.
- `branch` -- `:liquid` or `:vapour`.

### Keywords
- `p`  -- `(p_min, p_max)` absolute pressure range [Pa].
- `T`  -- `(T_min, T_max)` temperature range [K].
- `np`, `nT` -- number of grid nodes in each direction.
- `p_ref` -- evaluate every property at this single pressure, making them
             functions of temperature alone (see below). `nothing` (default)
             keeps the full two-dimensional dependence.
- `verbose` -- report the table extent and a few spot values when built.

### Locking the pressure (`p_ref`)

Passing `p_ref` builds properties that vary with temperature only, and - the
part that matters - sets the isothermal compressibility `psi` to **exactly
zero**, because a density that does not depend on pressure has no
compressibility to report. The two go together: freezing the lookup pressure
while still reporting the real `psi` would leave the pressure equation carrying a
compressibility the density does not have, which is worse than either extreme.

This is the right approximation whenever the pressure variation across the domain
is small compared with the temperature variation. For a heated pipe at 0.7 MPa
with a 618 Pa frictional drop and a few kelvin of wall superheat:

    pressure    : drho/rho = psi*dp   = 1.0e-7 * 618  = 0.006 %
    temperature : drho/rho = beta*dT  = 0.055  * 3    = 16 %

so temperature dominates by roughly three thousand to one, and locking the
pressure costs essentially nothing.

What it buys is stability. Near-critical liquid hydrogen has `psi ~ 1e-7` 1/Pa,
about 220x that of water, and the pressure equation's `psi*dp/dt` term is treated
with the time-step-start reference of `solve_pressure_compressible!` - written
for a sealed tank. In a flow-through domain, where the outlet already fixes the
pressure level, that term becomes a stiff spurious source. Setting `psi = 0`
removes it while keeping the thermal expansion `beta(T)` and the volume created
by phase change, both of which are real and needed.

Do **not** use `p_ref` where the pressure itself is the answer - a sealed
self-pressurising tank, say - since a zero compressibility cannot pressurise.

### Why a table

A direct Helmholtz evaluation Newton-solves for density and allocates roughly
twenty vectors per call, so it can neither run inside a `KernelAbstractions`
kernel nor be afforded per cell per time step. Tabulating once at setup makes
the real equation of state usable in the solver at the cost of an interpolation
error that is controllable through `np`/`nT`.

### The off-branch region

At a state where the requested branch does not exist as a stable root - vapour
properties well below `T_sat(p)`, or liquid properties well above it - the
branch is continued metastably where the root solve still converges, and
otherwise falls back to the value **on the saturation line at that pressure**.
This mirrors what `EOS_wrapper` already does, and it matters because the mixture
blend evaluates *both* phases in *every* cell: a single-phase liquid cell still
asks for a vapour density, and that request must return something finite and
physically sensible rather than fail.

### Example
```julia
# Saturated liquid hydrogen over the Tatsumoto pipe operating envelope
lh2 = RealFluid(H2(), :liquid, p=(0.30e6, 1.25e6), T=(20.0, 34.0))
gh2 = RealFluid(H2(), :vapour, p=(0.30e6, 1.25e6), T=(20.0, 34.0))

phases = (Phase(lh2), Phase(gh2))
```
"""
function RealFluid(
    fluid::HelmholtzEnergyFluid, branch::Symbol;
    p, T, np::Integer=81, nT::Integer=81, p_ref=nothing, verbose::Bool=true)

    branch in (:liquid, :vapour) || throw(ArgumentError(
        "`branch` must be :liquid or :vapour, got :$branch"))

    locked = p_ref !== nothing
    if locked
        p[1] <= p_ref <= p[2] || throw(ArgumentError(
            "`p_ref` = $p_ref lies outside the tabulated pressure range $(p)."))
    end

    # A pressure-locked table is constant along the pressure axis, so two nodes
    # carry it exactly and the lookup needs no special case.
    np_eff = locked ? 2 : np

    grid = PropertyGrid(p_min=p[1], p_max=p[2], np=np_eff, T_min=T[1], T_max=T[2], nT=nT)
    tables = build_property_tables(fluid, branch, grid; verbose=verbose, p_lock=p_ref)

    constants = helmholtz_constants(fluid, Float64)
    R_specific = constants.R_univ/constants.M

    # With the properties locked to one pressure, density genuinely no longer
    # depends on pressure, so the isothermal compressibility it presents to the
    # pressure equation must be ZERO. Returning the real psi here while holding
    # rho fixed would be worse than either extreme: the pressure equation would
    # carry a compressibility the density does not have.
    psi = locked ? zero(tables.psi) : tables.psi

    return RealFluidProperties(
        TabulatedEos(PropertyTable(grid, tables.rho),
                     PropertyTable(grid, psi), R_specific),
        TabulatedMu(PropertyTable(grid, tables.mu)),
        TabulatedK(PropertyTable(grid, tables.k)),
        TabulatedCp(PropertyTable(grid, tables.cp)),
        TabulatedBeta(PropertyTable(grid, tables.beta)),
    )
end

"""
    build_property_tables(fluid, branch, grid; verbose=true)

Evaluate `rho`, `psi`, `beta`, `cp`, `k` and `mu` at every node of `grid`,
returning them as a `NamedTuple` of `nT x np` matrices.

Separated from [`RealFluid`](@ref) so the raw arrays can be inspected or
compared against reference data in a test.
"""
function build_property_tables(
    fluid::HelmholtzEnergyFluid, branch::Symbol, grid::PropertyGrid;
    verbose::Bool=true, p_lock=nothing)

    constants = helmholtz_constants(fluid, Float64)
    ps = collect(grid_pressures(grid))
    Ts = collect(grid_temperatures(grid))
    np, nT = grid.np, grid.nT

    # `p_lock` evaluates every column at the same pressure, making the table
    # constant along the pressure axis. See the `p_ref` keyword of `RealFluid`.
    p_lock === nothing || (ps = fill(float(p_lock), np))

    rho  = Matrix{Float64}(undef, nT, np)
    psi  = Matrix{Float64}(undef, nT, np)
    beta = Matrix{Float64}(undef, nT, np)
    cp   = Matrix{Float64}(undef, nT, np)
    kth  = Matrix{Float64}(undef, nT, np)
    mu   = Matrix{Float64}(undef, nT, np)

    n_fallback = 0

    for (i, p) in enumerate(ps)
        # One saturation solve per pressure column rather than per node: it is
        # by far the most expensive call here (a secant loop wrapped around two
        # Newton density solves).
        T_sat = _saturation_temperature_or_nan(p, constants, fluid)

        # PASS 1: the root where the branch actually exists, NaN where it does not.
        col_rho = fill(NaN, nT)
        for (j, T) in enumerate(Ts)
            r = _branch_density_or_nan(fluid, constants, branch, p, T, T_sat)
            col_rho[j] = r === nothing ? NaN : r
        end

        # PASS 2: continue the branch into the off-branch region by HOLDING the
        # nearest valid node, rather than snapping to the saturation state.
        #
        # The saturation fallback this replaces was the source of a
        # non-monotonic density in every table this package has ever built. It
        # returned rho at T_sat, which belongs to a different temperature than
        # the neighbouring nodes, so the column stepped back UP mid-way - a
        # supersaturated vapour at 28 K is denser than saturated vapour at
        # 29.155 K, and the fallback put the lighter value next to the heavier
        # one. See `validate_property_table`.
        #
        # Holding the nearest valid node gives drho/dT = 0 across the off-branch
        # region: continuous, monotonic, and a defensible reading of "the branch
        # stops changing once it stops existing". The source temperature is
        # carried too, so the other five properties stay thermodynamically
        # consistent with the density they sit beside.
        src = _fill_offbranch!(col_rho)

        for j in 1:nT
            isfinite(col_rho[j]) || error(
                "No valid $(branch) root anywhere in the temperature column at " *
                "p = $p Pa. Narrow the pressure range, or check the branch exists here.")
            src[j] == j || (n_fallback += 1)
            props = _state_properties(fluid, constants, col_rho[j], Ts[src[j]])

            rho[j, i]  = props.rho
            psi[j, i]  = props.psi
            beta[j, i] = props.beta
            cp[j, i]   = props.cp
            kth[j, i]  = props.k
            mu[j, i]   = props.mu
        end
    end

    validate_property_table(rho, ps, Ts, branch, constants)

    if verbose
        @info """Tabulated $(typeof(fluid).name.name) $(branch) properties
        grid          : $(np) x $(nT) nodes  (p, T)
        pressure      : $(ps[1]/1e5) - $(ps[end]/1e5) bar
        temperature   : $(Ts[1]) - $(Ts[end]) K
        density       : $(round(minimum(rho), sigdigits=5)) - $(round(maximum(rho), sigdigits=5)) kg/m^3
        cp            : $(round(minimum(cp), sigdigits=5)) - $(round(maximum(cp), sigdigits=5)) J/kg/K
        off-branch    : $(n_fallback) of $(np*nT) nodes taken from the saturation line"""
    end

    return (rho=rho, psi=psi, beta=beta, cp=cp, k=kth, mu=mu)
end

"""
    validate_property_table(rho, ps, Ts, branch, constants)

Reject a density table that is discontinuous or non-monotonic in temperature.

**Why this exists.** A tabulated branch can be individually defensible at every
node and still be unusable as a whole. Building an H2 liquid table over
`T = (19, 40)` when `T_c = 33.145` produces exactly that: the metastable liquid
root is used up to ~31 K, the saturation-line fallback takes over around 32.5 K
(which makes `rho` *increase* with `T`), and above `T_c` there is only one
supercritical root, 9.2x lighter. Each step follows the rules in
[`_branch_density`](@ref); the composite is a cliff.

A solver cannot survive that. The near-wall liquid heats through `T_sat` by
construction in a boiling case, and a 9x density drop across one cell is a
discontinuous `grad(rho)` straight into the momentum and pressure equations. It
took a long time to find because it is invisible from outside: `table_range_report`
confirms the run stayed *inside* the table bounds, and it did - the table was
wrong *within* its declared range.

So this fails at BUILD time, where the message can name the offending state and
the remedy, rather than at run time as an unexplained divergence.

`d(rho)/dT` must be negative (a fluid expands when heated at fixed pressure) and
must not change by more than `_MAX_DRHO_JUMP` between adjacent nodes.
"""
# A physical branch thins smoothly with temperature. These bounds are loose - the
# failure they catch is a factor-of-nine cliff, not a percent-level wobble - so a
# genuine near-critical steepening will not trip them.
const _MAX_DRHO_RATIO = 3.0     # max |rho[j+1]/rho[j]| step between adjacent nodes
const _MONOTONIC_TOL  = 1e-8    # allow round-off-sized positive drho/dT

function validate_property_table(rho, ps, Ts, branch, constants)
    nT, np = size(rho)
    nT < 2 && return nothing

    for i in 1:np, j in 1:(nT - 1)
        r1, r2 = rho[j, i], rho[j + 1, i]
        (isfinite(r1) && isfinite(r2) && r1 > 0 && r2 > 0) || throw(ArgumentError(
            """Property table for the $(branch) branch is not finite at \
p = $(round(ps[i]/1e5, digits=4)) bar, T = $(round(Ts[j+1], digits=4)) K \
(rho = $r2 kg/m^3)."""))

        ratio = max(r1/r2, r2/r1)
        if ratio > _MAX_DRHO_RATIO
            throw(ArgumentError(
                """Property table for the $(branch) branch is DISCONTINUOUS.

  density jumps $(round(ratio, digits=2))x between adjacent temperature nodes:
      T = $(round(Ts[j], digits=4)) K  ->  rho = $(round(r1, digits=5)) kg/m^3
      T = $(round(Ts[j+1], digits=4)) K  ->  rho = $(round(r2, digits=5)) kg/m^3
  at p = $(round(ps[i]/1e5, digits=4)) bar.

  The usual cause is a temperature range that crosses the critical point:
  T_c = $(round(constants.T_c, digits=4)) K for this fluid. Above T_c there is no
  liquid branch - both branches collapse onto the single supercritical root - so
  a table spanning T_c contains a step change no solver can survive.

  Cap the table's temperature range below T_c, and clamp the solver's own limits
  to match (e.g. the `limit` keyword of the temperature `SolverSetup`)."""))
        end

        if (r2 - r1) > _MONOTONIC_TOL*max(r1, r2)
            throw(ArgumentError(
                """Property table for the $(branch) branch is NON-MONOTONIC.

  density RISES with temperature at constant pressure:
      T = $(round(Ts[j], digits=4)) K  ->  rho = $(round(r1, digits=5)) kg/m^3
      T = $(round(Ts[j+1], digits=4)) K  ->  rho = $(round(r2, digits=5)) kg/m^3
  at p = $(round(ps[i]/1e5, digits=4)) bar.

  A fluid must expand when heated at fixed pressure, so this is not physical. It
  usually means the metastable continuation ran out and the saturation-line
  fallback took over mid-column, which steps the density back up to its saturated
  value (see `_branch_density`).

  Narrow the table's temperature range towards the saturation temperature at this
  pressure, or reduce `_METASTABLE_BAND`."""))
        end
    end
    return nothing
end

"""
`T_sat(p)`, or `NaN` at or above the critical pressure where there is no
saturation line to find.
"""
function _saturation_temperature_or_nan(p, constants, fluid)
    p >= constants.p_c && return NaN
    p <= constants.p_triple && return NaN
    return try
        find_saturation_temperature(p, constants, fluid)
    catch
        NaN
    end
end

"""
    _branch_density(fluid, constants, branch, p, T, T_sat) -> (rho_molar, fell_back)

Molar density of the requested branch at `(p, T)`.

`fell_back` is `1` when the state had to be taken from the saturation line
because the branch has no root at `T` (see the `RealFluid` docstring), and `0`
when the root at `T` itself was used - including a metastable one.
"""
# Temperature margin, in K, over which a branch is continued past its saturation
# temperature before the saturated state is used instead.
#
# A liquid a few kelvin above T_sat is metastable but perfectly well defined by
# the equation of state, and it is the state that actually exists in the
# superheated layer next to a heated wall - so continuing the branch there gives
# a smoother and more physical beta and cp than freezing at saturation.
#
# Far beyond that margin the branch has no root at all: the density solve then
# wanders, burns its full iteration budget and returns nothing useful. Not
# attempting it is both faster and quieter than catching the failure.
const _METASTABLE_BAND = 5.0

function _branch_density(fluid, constants, branch, p, T, T_sat)
    (; rho_c, R_univ, liquid_multiplier) = constants

    guess(Tg) = branch === :liquid ? liquid_multiplier*rho_c : p/(R_univ*Tg)

    # Supercritical in either variable: one root, no branches to distinguish.
    if isnan(T_sat) || T >= constants.T_c
        rho = _try_density(T, p, guess(T), constants, fluid)
        rho === nothing || return (rho, 0)
        # A single stubborn node should not abort a whole table build; the
        # opposite guess is the only other sensible starting point.
        rho = _try_density(T, p, branch === :liquid ? p/(R_univ*T) : liquid_multiplier*rho_c,
                           constants, fluid)
        rho === nothing && error(
            "Helmholtz density solve failed at p = $p Pa, T = $T K ($(branch) branch). " *
            "Narrow the table range, or check that the state is within the EOS validity domain.")
        return (rho, 0)
    end

    within_band = branch === :liquid ? (T <= T_sat + _METASTABLE_BAND) :
                                       (T >= T_sat - _METASTABLE_BAND)

    if within_band
        rho = _try_density(T, p, guess(T), constants, fluid)
        if rho !== nothing && _on_branch(branch, rho, rho_c)
            return (rho, 0)
        end
    end

    # No root on this branch at T (or the solve wandered onto the other one):
    # take the saturated state at this pressure.
    rho_sat = _try_density(T_sat, p, guess(T_sat), constants, fluid)
    rho_sat === nothing && error(
        "Helmholtz density solve failed on the saturation line at p = $p Pa " *
        "(T_sat = $T_sat K, $(branch) branch).")
    return (rho_sat, 1)
end

"""
    _branch_density_or_nan(fluid, constants, branch, p, T, T_sat) -> rho_molar or nothing

The requested branch's molar density at `(p, T)`, or `nothing` where that branch
has no root - metastable continuation included, saturation fallback NOT.

Split out from [`_branch_density`](@ref) so the off-branch region can be filled
by continuation across the whole column (see `_fill_offbranch!`) rather than
node-by-node, which is what makes the result monotonic.
"""
function _branch_density_or_nan(fluid, constants, branch, p, T, T_sat)
    (; rho_c, R_univ, liquid_multiplier) = constants
    guess(Tg) = branch === :liquid ? liquid_multiplier*rho_c : p/(R_univ*Tg)

    # Supercritical in either variable: one root, no branches to distinguish.
    if isnan(T_sat) || T >= constants.T_c
        rho = _try_density(T, p, guess(T), constants, fluid)
        rho === nothing || return rho
        return _try_density(T, p, branch === :liquid ? p/(R_univ*T) : liquid_multiplier*rho_c,
                            constants, fluid)
    end

    within_band = branch === :liquid ? (T <= T_sat + _METASTABLE_BAND) :
                                       (T >= T_sat - _METASTABLE_BAND)
    within_band || return nothing

    rho = _try_density(T, p, guess(T), constants, fluid)
    (rho !== nothing && _on_branch(branch, rho, rho_c)) ? rho : nothing
end

"""
    _fill_offbranch!(col) -> src

Fill the `NaN` entries of a temperature column by holding the nearest valid
neighbour, returning for each node the index it took its value from (itself,
where the node was valid).

Leading `NaN`s take the first valid node and trailing ones the last, so the
column is continuous at both ends and flat wherever the branch does not exist.
`src` lets the caller evaluate the remaining properties at the SOURCE
temperature, keeping each node internally consistent.
"""
function _fill_offbranch!(col)
    n = length(col)
    src = collect(1:n)
    first_valid = findfirst(isfinite, col)
    first_valid === nothing && return src      # caller reports the empty column

    # backwards from the first valid node, then forwards from every other gap
    for j in 1:(first_valid - 1)
        col[j] = col[first_valid]; src[j] = first_valid
    end
    for j in (first_valid + 1):n
        if !isfinite(col[j])
            col[j] = col[j - 1]; src[j] = src[j - 1]
        end
    end
    return src
end

# Below the critical temperature the two roots sit either side of the critical
# density, which is a robust way to tell which one the solver landed on.
_on_branch(branch, rho_mol, rho_c) =
    branch === :liquid ? rho_mol > rho_c : rho_mol < rho_c

function _try_density(T, p, rho_guess, constants, fluid)
    # `find_density_advanced` reports non-convergence with `@warn` and returns
    # nothing. Failure is an expected, handled outcome here - the caller falls
    # back to the saturation line and the count is reported at the end of the
    # build - so the warning would be noise, not information.
    # `Base.CoreLogging` rather than the `Logging` stdlib, which is not a
    # declared dependency of this package.
    return Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        try
            rho = find_density_advanced(T, p, rho_guess, constants, fluid)
            (rho === nothing || !isfinite(rho) || rho <= 0) ? nothing : rho
        catch
            nothing
        end
    end
end

"""
    _state_properties(fluid, constants, rho_molar, T)

All six tabulated properties at a single fully-determined state, in SI mass
units. See the unit notes at the top of this file.
"""
function _state_properties(fluid, constants, rho_mol, T)
    (; T_c, rho_c, M, T_ref) = constants

    tau = T_c/T
    delta = rho_mol/rho_c

    rho_mass = rho_mol*M

    cp_mass = c_p(delta, tau, constants, fluid)/M
    cv_mass = c_v(delta, tau, constants, fluid)/M

    # `k_T` is the isothermal compressibility (1/rho)(drho/dp)|_T in 1/Pa, which
    # is exactly the coefficient the pressure equation needs. It is also what
    # the thermal conductivity model wants for its critical enhancement, so the
    # one call serves both.
    psi = k_T(T, rho_mol, constants, fluid)
    kT_ref = k_T(T_ref, rho_mol, constants, fluid)

    beta = beta_calc(T, delta, tau, constants, fluid)

    # The viscosity and conductivity correlations are written in terms of MASS
    # density, and viscosity is returned in micro-Pa s.
    mu_micro = _mu_high_fidelity(fluid, T, rho_mass)
    k_cond = _thermal_conductivity(fluid, rho_mass, T, cp_mass, cv_mass, psi, kT_ref, mu_micro)

    return (rho=rho_mass, psi=psi, beta=beta, cp=cp_mass,
            k=k_cond, mu=mu_micro*1.0e-6)
end


"""
    phase_properties_at(fluid, p, T; branch=:liquid) -> NamedTuple

Evaluate the Helmholtz equation of state at a **single** state, returning
`(rho, psi, beta, cp, k, mu)` in SI mass units.

Intended for setting *constant* phase properties at a known operating point,
where tabulation would be pointless:

```julia
sat = phase_properties_at(H2(), 0.7e6, 29.15, branch = :liquid)
liquid = Phase(rho = sat.rho, mu = sat.mu, k = sat.k, cp = sat.cp, beta = sat.beta)
```

Taking the numbers from the EOS rather than hardcoding them keeps a
constant-property phase consistent with the tabulated one it sits next to, and
means the two cannot silently drift apart.

Follows the same off-branch handling as [`RealFluid`](@ref): where the requested
branch has no root at `T`, the value on the saturation line at `p` is returned.
"""
function phase_properties_at(fluid::HelmholtzEnergyFluid, p, T; branch::Symbol=:liquid)
    branch in (:liquid, :vapour) || throw(ArgumentError(
        "`branch` must be :liquid or :vapour, got :$branch"))

    constants = helmholtz_constants(fluid, Float64)
    T_sat = _saturation_temperature_or_nan(float(p), constants, fluid)
    rho_mol, fell_back = _branch_density(fluid, constants, branch, float(p), float(T), T_sat)
    T_eval = fell_back == 1 ? T_sat : float(T)

    return _state_properties(fluid, constants, rho_mol, T_eval)
end


# =============================================================================
#  Saturation curve
# =============================================================================

"""
    build_saturation_curve(fluid; p, T, np=201, nT=201, verbose=true)

Tabulate `T_sat(p)`, `h_fg(p)` and `p_sat(T)` from the Helmholtz equation of
state, returning a [`SaturationCurve`](@ref) to pass as
`Fluid{Multiphase}(..., saturation = ...)`.

The alternative, [`Antoine`](@ref), is a fit valid over a stated temperature
window and carries no latent heat. This is preferable whenever the run spans a
wide pressure range or approaches the critical point, where `h_fg` collapses
towards zero and a constant value stops being meaningful.

### Example
```julia
sat = build_saturation_curve(H2(), p=(0.30e6, 1.25e6), T=(20.0, 33.0))
h_fg_op = latent_heat(sat, 0.7e6, 0.0)     # value at the operating pressure
```
"""
function build_saturation_curve(
    fluid::HelmholtzEnergyFluid; p, T, np::Integer=201, nT::Integer=201, verbose::Bool=true)

    constants = helmholtz_constants(fluid, Float64)
    (; M) = constants

    ps = range(float(p[1]), float(p[2]), length=np)
    Ts = range(float(T[1]), float(T[2]), length=nT)

    T_sat_tab = Vector{Float64}(undef, np)
    h_fg_tab = Vector{Float64}(undef, np)
    p_sat_tab = Vector{Float64}(undef, nT)

    for (i, pi) in enumerate(ps)
        T_sat = _saturation_temperature_or_nan(pi, constants, fluid)
        if isnan(T_sat)
            # At or above the critical pressure the two phases are
            # indistinguishable: T_sat saturates at T_c and the latent heat is
            # zero. Both are the correct limits, and both keep the table
            # monotonic and finite so a lookup near p_c stays well behaved.
            T_sat_tab[i] = constants.T_c
            h_fg_tab[i] = 0.0
            continue
        end

        T_sat_tab[i] = T_sat

        (_, _, rho_l, rho_v) = find_saturation_properties(T_sat, pi, constants, fluid)
        tau = constants.T_c/T_sat
        h_l = enthalpy_calc(T_sat, rho_l/constants.rho_c, tau, constants, fluid)/M
        h_v = enthalpy_calc(T_sat, rho_v/constants.rho_c, tau, constants, fluid)/M
        h_fg_tab[i] = h_v - h_l
    end

    for (j, Tj) in enumerate(Ts)
        p_sat_tab[j] = if Tj >= constants.T_c
            constants.p_c
        else
            try
                # `find_saturation_properties` returns the converged p_sat; the
                # `pressure` argument only steers the T_sat it also computes,
                # which is discarded here.
                find_saturation_properties(Tj, constants.p_c, constants, fluid)[1]
            catch
                vapour_pressure_ancillary(Tj, constants, fluid)
            end
        end
    end

    if verbose
        @info """Tabulated $(typeof(fluid).name.name) saturation curve
        pressure    : $(ps[1]/1e5) - $(ps[end]/1e5) bar  ($(np) nodes)
        T_sat       : $(round(T_sat_tab[1], digits=3)) - $(round(T_sat_tab[end], digits=3)) K
        h_fg        : $(round(h_fg_tab[end]/1e3, digits=2)) - $(round(h_fg_tab[1]/1e3, digits=2)) kJ/kg
        temperature : $(Ts[1]) - $(Ts[end]) K  ($(nT) nodes)"""
    end

    dp = np > 1 ? (ps[end] - ps[1])/(np - 1) : one(Float64)
    dT = nT > 1 ? (Ts[end] - Ts[1])/(nT - 1) : one(Float64)

    return SaturationCurve(
        float(ps[1]), dp, np,
        float(Ts[1]), dT, nT,
        T_sat_tab, h_fg_tab, p_sat_tab)
end


# =============================================================================
#  Range diagnostics
# =============================================================================

"""
    table_range_report(eos::TabulatedEos, p_abs, T) -> NamedTuple

Report how much of the current solution lies outside the tabulated `(p, T)`
range, where lookups are frozen at the boundary value rather than extrapolated.

Intended as a post-run (or per-write-interval) check: a non-zero count means the
table is too narrow for the case and the affected cells are carrying boundary
values, which is a silent accuracy loss rather than a visible failure.

Returns the fraction and count of out-of-range cells and the observed extremes.
"""
function table_range_report(eos::TabulatedEos, p_abs, T)
    grid = eos.rho.grid
    p_lo, p_hi = grid.p_min, grid_p_max(grid)
    T_lo, T_hi = grid.T_min, grid_T_max(grid)

    pv = Array(p_abs.values)
    Tv = Array(T.values)

    outside = count(i -> pv[i] < p_lo || pv[i] > p_hi || Tv[i] < T_lo || Tv[i] > T_hi,
                    eachindex(pv))

    return (
        outside = outside,
        total = length(pv),
        fraction = outside/length(pv),
        p_range = (minimum(pv), maximum(pv)),
        p_table = (p_lo, p_hi),
        T_range = (minimum(Tv), maximum(Tv)),
        T_table = (T_lo, T_hi),
    )
end
