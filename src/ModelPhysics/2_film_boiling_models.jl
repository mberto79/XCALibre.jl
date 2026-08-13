export AbstractCriticalHeatFlux, Zuber, BubbleCrowding, FixedCriticalHeatFlux
export AbstractMinimumFilmBoiling, Berenson, HomogeneousNucleation
export FixedMinimumFilmBoiling
export AbstractFilmBoilingHTC, ForcedConvectionFilm, Bromley
export FilmBoiling, FilmClosure
export AbstractTransitionDriver, SuperheatTransition, VoidTransition
export AbstractVoidMeasure, NearWallCell, BubblyLayerAverage
export critical_heat_flux, minimum_film_superheat, film_boiling_htc
export film_boiling_fraction, film_closure, superheat_at_flux
export void_layer_thickness, void_layer_thickness_for
export needs_layer_average, wall_void_fraction, recorded_void_fraction

# =============================================================================
#  Departure from nucleate boiling and the film boiling regime
# =============================================================================
#
#  RPI closes the wall flux with a LIQUID-WETTED wall. Past critical heat flux
#  that premise fails: the wall is blanketed and heat crosses a vapour film. This
#  file supplies the three pieces needed to carry a case through that transition
#  and out the other side.
#
#      1. WHERE nucleate boiling ends  -> `AbstractCriticalHeatFlux`
#      2. WHERE film boiling is fully established -> `AbstractMinimumFilmBoiling`
#      3. WHAT the wall flux is once it is -> `AbstractFilmBoilingHTC`
#
#  ---------------------------------------------------------------------------
#  WHY BLEND RATHER THAN SWITCH
#  ---------------------------------------------------------------------------
#
#  The obvious implementation - "if q > q_CHF, use the film model" - is not
#  stable under a PRESCRIBED wall flux, which is what this solver imposes. The
#  film model needs far more superheat to pass the same flux, so the switch
#  raises `T_w`; but the test that fired was on flux, which has not changed, so
#  nothing prevents the next iterate switching straight back. The wall then
#  chatters between two branches with no fixed point in between.
#
#  Instead the two closures are BLENDED into a single continuous `q_w(T_w)`
#  spanning nucleate, transition and film boiling. The blend runs on WALL
#  SUPERHEAT, between two superheats obtained from correlations:
#
#      dT_lo  the superheat at which the RPI partition delivers `q_CHF`
#      dT_hi  the minimum film boiling (Leidenfrost) superheat
#
#  `dT_lo` needs no correlation of its own: `q_RPI(dT)` is monotone, so inverting
#  the already-calibrated partition at `q_CHF` gives the matching superheat
#  directly. Only `q_CHF` and `dT_hi` are new closures, and both are chosen here
#  to be PROPERTY-ONLY so they carry to other fluids without refitting.
#
#  ---------------------------------------------------------------------------
#  WHY THE WALL MUST HAVE THERMAL CAPACITY
#  ---------------------------------------------------------------------------
#
#  Across the transition region `q_w(T_w)` is DECREASING - that is the physical
#  content of the boiling curve, and it is why DNB is a jump. A flux-controlled
#  algebraic inversion of a non-monotone curve has up to THREE roots, and
#  bisection will return whichever one the bracket happens to straddle, silently.
#
#  Integrating the wall energy balance instead has no such ambiguity: the wall
#  FOLLOWS the curve, and the jump at DNB emerges as a fast transient rather than
#  a root selection. `FilmBoiling` therefore requires `RPI(wall_capacity > 0)`
#  and the constructor rejects anything else - see the error text there.
# =============================================================================


# =============================================================================
#  Critical heat flux
# =============================================================================

abstract type AbstractCriticalHeatFlux end

"""
    critical_heat_flux(model, state) -> q_CHF [W/m^2]

Wall heat flux at which nucleate boiling departs.
"""
function critical_heat_flux end

"""
    Zuber(; C=0.131)

Zuber (1959) hydrodynamic critical heat flux.

    q_CHF = C h_fg sqrt(rho_v) [sigma g (rho_l - rho_v)]^(1/4)

The limit is set by Helmholtz instability of the vapour columns leaving the
surface, so it is built entirely from fluid properties with no fitted constant
tied to a working fluid - `C = pi/24 ~ 0.131` comes out of the stability
analysis. That is exactly the property wanted here: it transfers to hydrogen,
nitrogen or water without recalibration.

### What it does not include

It is a POOL boiling result. Forced flow removes vapour from the surface and
raises CHF, typically by a factor of 1.5-3 at the mass fluxes in a heated tube,
and subcooling raises it further. Zuber is therefore a CONSERVATIVE floor rather
than a prediction for flow boiling: it will make the model depart from nucleate
boiling too early.

For quantitative flow boiling work use a tube-level correlation - Katto & Ohno
(1984) or Shah (1987), both validated across cryogens - and pass the resulting
value through [`FixedCriticalHeatFlux`](@ref). Those correlations depend on mass
flux, heated length, diameter and inlet quality, none of which are local
quantities, so a scalar is the correct interface for them rather than a
limitation of this one.

`C` is exposed so the pool value can be scaled by a flow multiplier if one is
known for the geometry.
"""
struct Zuber{F<:AbstractFloat} <: AbstractCriticalHeatFlux
    C::F
end
Zuber(; C = 0.131) = Zuber(float(C))
Adapt.@adapt_structure Zuber

@inline function critical_heat_flux(model::Zuber, s::BoilingState{F}) where F
    drho = s.rho_l - s.rho_v
    # Every ingredient must be physically present for the correlation to mean
    # anything. `Inf` is the SAFE degenerate value - it says "no CHF limit here",
    # which leaves the nucleate branch untouched.
    #
    # Returning zero instead would be catastrophic and silent: `q_CHF` caps the
    # nucleate partition, so a zero cap annihilates the ENTIRE wall heat flux at
    # every wall temperature, the transient balance collapses to
    # `C dT_w/dt = q_gen`, and the wall temperature runs away linearly until the
    # bisection bracket overflows. That is exactly what `rho_v = 0` produced on
    # the LH2 pipe - the phase density is zero until the solver's first property
    # update, and the wall boiling pass sees it before then.
    (drho <= zero(F) ||          # supercritical: no phase separation
     s.rho_v <= zero(F) ||       # no vapour present (or not yet initialised)
     s.sigma <= zero(F) ||       # no surface tension supplied
     s.h_fg <= zero(F)) && return F(Inf)
    return F(model.C)*s.h_fg*sqrt(s.rho_v)*(s.sigma*s.g*drho)^F(0.25)
end

"""
    BubbleCrowding(; A_crit=1.0)

Critical heat flux taken from the RPI model's OWN bubble coverage: departure
occurs when the bubble influence area fraction `A_b` reaches `A_crit`.

This is not a correlation at all, which is its attraction - it introduces no new
empirical constants and is automatically consistent with whatever site density
and departure diameter the case has been calibrated with. Physically it is the
bubble-crowding picture of DNB: once neighbouring bubble influence areas overlap
the whole surface, liquid can no longer reach it.

### The trade

It inherits the site-density calibration entirely. `A_b` is built from
`N_a * D_d^2`, and `N_a` is the least certain object in the model - a power law
fitted over a limited flux range. Where that fit is poor, the predicted CHF is
poor in the same way, and it will not be obvious which is at fault.

Use it as a CROSS-CHECK against [`Zuber`](@ref): agreement between a
property-only correlation and the model's own coverage limit is real evidence,
whereas either alone is not.
"""
struct BubbleCrowding{F<:AbstractFloat} <: AbstractCriticalHeatFlux
    A_crit::F
end
BubbleCrowding(; A_crit = 1.0) = BubbleCrowding(float(A_crit))
Adapt.@adapt_structure BubbleCrowding

"""
    FixedCriticalHeatFlux(q)

A CHF value supplied directly [W/m^2].

This is the intended route for the tube-level flow boiling correlations
(Katto & Ohno, Shah, Bowring): they are functions of mass flux, heated length,
diameter and inlet subcooling, so they are evaluated once for the case and
handed in, not recomputed per face.
"""
struct FixedCriticalHeatFlux{F<:AbstractFloat} <: AbstractCriticalHeatFlux
    q::F
end
FixedCriticalHeatFlux(; q) = FixedCriticalHeatFlux(float(q))
Adapt.@adapt_structure FixedCriticalHeatFlux

@inline critical_heat_flux(model::FixedCriticalHeatFlux, s::BoilingState{F}) where F =
    F(model.q)


# =============================================================================
#  Minimum film boiling (Leidenfrost) temperature
# =============================================================================

abstract type AbstractMinimumFilmBoiling end

"""
    minimum_film_superheat(model, state) -> dT_min [K]

Wall superheat at which a stable vapour film is established - the lower end of
the film boiling branch, and the upper end of the transition.
"""
function minimum_film_superheat end

"""
    Berenson(; C=0.127)

Berenson (1961) minimum film boiling superheat, from Taylor instability of the
vapour-liquid interface under the film:

    dT_min = C (rho_v h_fg / k_v)
             [g drho/(rho_l + rho_v)]^(2/3)
             [sigma/(g drho)]^(1/2)
             [mu_v/(g drho)]^(1/3)

Property-only, so it transfers between fluids. It is the standard choice and the
one to reach for first.

### Caveats worth knowing before trusting it

It is a POOL boiling, flat-plate result derived for a film collapsing under its
own instability. Forced flow destabilises the film and LOWERS `dT_min`, often
substantially. It also takes no account of the wall material: a low-effusivity
wall rewets at a lower superheat than a high-effusivity one (Henry's 1974
correction addresses this), and at cryogenic temperature the effusivity of steel
is far from its room-temperature value because `cp` collapses as `T^3`.

Requires `k_v` and `mu_v` on the state; returns zero if they are unset.
"""
struct Berenson{F<:AbstractFloat} <: AbstractMinimumFilmBoiling
    C::F
end
Berenson(; C = 0.127) = Berenson(float(C))
Adapt.@adapt_structure Berenson

@inline function minimum_film_superheat(model::Berenson, s::BoilingState{F}) where F
    drho = s.rho_l - s.rho_v
    (drho <= zero(F) || s.k_v <= zero(F) || s.mu_v <= zero(F)) && return zero(F)
    gd = s.g*drho
    return F(model.C)*(s.rho_v*s.h_fg/s.k_v)*
           (gd/(s.rho_l + s.rho_v))^(F(2)/3)*
           sqrt(s.sigma/gd)*
           (s.mu_v/gd)^(F(1)/3)
end

"""
    HomogeneousNucleation(; T_crit, C=0.9)

Minimum film boiling superheat from the THERMODYNAMIC limit of liquid
superheat rather than from film hydrodynamics:

    dT_min = C T_crit - T_sat

Liquid in contact with a wall hotter than the homogeneous nucleation temperature
flashes on contact, so a vapour film cannot collapse above it however unstable it
is. That makes this an upper bound on `dT_min` which no hydrodynamic argument can
exceed, and it needs one property - the critical temperature.

### Why this matters for cryogens specifically

The bound BINDS for low-critical-temperature fluids in a way it does not for
water. Hydrogen has `T_crit = 33.15 K`, so at 4 bar (`T_sat ~ 26 K`) the whole
available superheat before the thermodynamic limit is only a few kelvin - the
entire boiling curve, nucleate through film, is compressed into a range that for
water would be a rounding error. A correlation like [`Berenson`](@ref) fitted at
water-like scales can easily return a `dT_min` exceeding the limit entirely, at
which point it is not describing a reachable state.

`C` between 0.84 (van der Waals spinodal, `27/32 T_crit`) and 0.9 (the value
commonly used with real fluids) covers the usual range; the default is the
latter. `T_crit` is in kelvin and must be supplied.

Taking `min(Berenson, HomogeneousNucleation)` is the recommended combination and
is what [`FilmBoiling`](@ref) does when given both.
"""
struct HomogeneousNucleation{F<:AbstractFloat} <: AbstractMinimumFilmBoiling
    T_crit::F
    C::F
end
function HomogeneousNucleation(; T_crit, C = 0.9)
    T_crit > 0 || throw(ArgumentError(
        "`T_crit` must be positive [K], got $T_crit"))
    0 < C <= 1 || throw(ArgumentError(
        "`C` must be in (0, 1] - it is a fraction of the critical temperature, got $C"))
    return HomogeneousNucleation(float(T_crit), float(C))
end
Adapt.@adapt_structure HomogeneousNucleation

@inline minimum_film_superheat(model::HomogeneousNucleation, s::BoilingState{F}) where F =
    max(F(model.C)*F(model.T_crit) - s.T_sat, zero(F))

"""
    FixedMinimumFilmBoiling(dT)

A minimum film boiling superheat supplied directly [K].
"""
struct FixedMinimumFilmBoiling{F<:AbstractFloat} <: AbstractMinimumFilmBoiling
    dT::F
end
FixedMinimumFilmBoiling(; dT) = FixedMinimumFilmBoiling(float(dT))
Adapt.@adapt_structure FixedMinimumFilmBoiling

@inline minimum_film_superheat(model::FixedMinimumFilmBoiling, s::BoilingState{F}) where F =
    F(model.dT)

# Combining two closures takes the smaller: a film cannot be more stable than
# either mechanism allows. This is what makes `min(Berenson, homogeneous limit)`
# expressible without a special case.
struct MinOfTwo{A,B} <: AbstractMinimumFilmBoiling
    a::A
    b::B
end
Adapt.@adapt_structure MinOfTwo

@inline minimum_film_superheat(m::MinOfTwo, s::BoilingState{F}) where F =
    min(minimum_film_superheat(m.a, s), minimum_film_superheat(m.b, s))


# =============================================================================
#  Film boiling heat transfer
# =============================================================================

abstract type AbstractFilmBoilingHTC end

"""
    film_boiling_htc(model, state, y_plus, u_tau) -> h_f [W/m^2/K]

Heat transfer coefficient across an established vapour film.
"""
function film_boiling_htc end

"""
    ForcedConvectionFilm(; Pr_t=0.85)

Film boiling as single-phase forced convection **in the vapour**: the same
thermal law of the wall used for the liquid side, evaluated with vapour
properties.

    h_f = rho_v cp_v u_tau / T+(Pr_v)

### Why this rather than Bromley

[`Bromley`](@ref) is a laminar free-convection result for a film rising under
buoyancy, and it is the right model only when buoyancy dominates. The standard
criterion (Bromley, LeRoy & Robbers) is the tube Froude number: free convection
applies below `U ~ sqrt(g D)` and forced convection above `2 sqrt(g D)`.

For a 6 mm tube that threshold is `sqrt(9.81*0.006) ~ 0.24 m/s`. A flow boiling
case running at several m/s is an order of magnitude clear of it, so the forced
convection form is not merely acceptable there, it is the correct one and
Bromley would be the approximation.

It also reuses [`single_phase_htc`](@ref) unchanged, so the film branch inherits
a routine already exercised on the liquid side rather than introducing a second
untested correlation.

Requires `cp_v`, `k_v` and `mu_v` on the state; returns zero if they are unset.
"""
struct ForcedConvectionFilm{F<:AbstractFloat} <: AbstractFilmBoilingHTC
    Pr_t::F
end
ForcedConvectionFilm(; Pr_t = 0.85) = ForcedConvectionFilm(float(Pr_t))
Adapt.@adapt_structure ForcedConvectionFilm

@inline function film_boiling_htc(
    model::ForcedConvectionFilm, s::BoilingState{F}, y_plus, u_tau) where F
    (s.k_v <= zero(F) || s.cp_v <= zero(F) || s.mu_v <= zero(F)) && return zero(F)

    # y+ was formed with the LIQUID viscosity by the caller. The film is vapour,
    # so the wall distance in viscous units is different by nu_l/nu_v - a large
    # factor, since vapour is both lighter and less viscous. Rescaling here keeps
    # `single_phase_htc` on the branch of the law of the wall that the vapour
    # film actually occupies.
    nu_l = s.mu_l/max(s.rho_l, eps(F))
    nu_v = s.mu_v/max(s.rho_v, eps(F))
    yp_v = y_plus*nu_l/max(nu_v, eps(F))

    return single_phase_htc(yp_v, u_tau, s.rho_v, s.cp_v, s.mu_v, s.k_v, F(model.Pr_t))
end

"""
    Bromley(; C=0.62, D)

Bromley (1950) laminar film boiling on a horizontal cylinder of diameter `D`:

    h_f = C [k_v^3 rho_v drho g h_fg' / (D mu_v dT_sup)]^(1/4)

with `h_fg' = h_fg (1 + 0.4 cp_v dT_sup/h_fg)` correcting for sensible heating
of the vapour crossing the film.

A free-convection result - see the Froude criterion in
[`ForcedConvectionFilm`](@ref) for when it applies. Retained because it is the
reference film boiling correlation and a useful check on the forced convection
form at low velocity; for a pumped loop it will UNDER-predict.

Radiation across the film is not included. That is a deliberate omission for
cryogenic work - at a film boiling wall temperature of order 100 K the
`sigma T^4` contribution is negligible beside convection - but it is NOT safe at
the several-hundred-kelvin wall temperatures reached in water systems.
"""
struct Bromley{F<:AbstractFloat} <: AbstractFilmBoilingHTC
    C::F
    D::F
end
function Bromley(; C = 0.62, D)
    D > 0 || throw(ArgumentError("`D` must be a positive length scale [m], got $D"))
    return Bromley(float(C), float(D))
end
Adapt.@adapt_structure Bromley

@inline function film_boiling_htc(model::Bromley, s::BoilingState{F}, y_plus, u_tau) where F
    drho = s.rho_l - s.rho_v
    dT = s.dT_sup
    (drho <= zero(F) || dT <= zero(F) || s.k_v <= zero(F) || s.mu_v <= zero(F)) &&
        return zero(F)

    h_fg_eff = s.h_fg*(one(F) + F(0.4)*s.cp_v*dT/s.h_fg)
    return F(model.C)*(s.k_v^3*s.rho_v*drho*s.g*h_fg_eff/
                       (F(model.D)*s.mu_v*dT))^F(0.25)
end


# =============================================================================
#  The film boiling closure
# =============================================================================

# =============================================================================
#  What drives the transition
# =============================================================================

abstract type AbstractTransitionDriver end

"""
    SuperheatTransition()

Blend from nucleate to film boiling on WALL SUPERHEAT, between the superheat at
which the RPI partition delivers `q_CHF` and the minimum film boiling superheat.
The default, and the formulation described in [`FilmBoiling`](@ref).

Needs a critical heat flux value. That is its limitation: `Zuber` came out +52%
and `BubbleCrowding` -43% against measurement on the LH2 pipe, so in practice it
wants either a measured CHF or a tube-level correlation through
[`FixedCriticalHeatFlux`](@ref) - neither of which is available for an arbitrary
new case.
"""
struct SuperheatTransition <: AbstractTransitionDriver end
Adapt.@adapt_structure SuperheatTransition

"""
    VoidTransition(; measure=NearWallCell(), alpha_1=0.8, alpha_2=0.95)

Blend on NEAR-WALL VAPOUR FRACTION instead of superheat:

    w = smoothstep((alpha_v - alpha_1)/(alpha_2 - alpha_1))

Departure then EMERGES from the solution rather than being located by a
correlation - vapour accumulates at the wall until liquid can no longer reach it.
This is the bubble-crowding picture of DNB (Weisman & Pei put the critical
bubbly-layer void near 0.82) and it is what the wall boiling regime maps in the
commercial codes key off.

### What it buys

No critical heat flux value is needed to locate the transition, so the model can
be applied to a geometry or fluid with no measured CHF. It also predicts WHERE
departure starts - wherever void accumulates first - which a flux criterion
cannot, since flux is imposed uniformly.

### What it still needs

`FilmBoiling` keeps its `chf` field, but with this driver it serves only to CAP
the nucleate branch, not to locate the transition. Capping is still required:
`q_evap ~ dT_sup^n` will otherwise run away faster than the blend can suppress it
(see the four-argument [`wall_heat_partition`](@ref)). A cap is an upper bound
rather than a prediction, so a property-only [`Zuber`](@ref) is adequate there
even when it is a poor CHF estimate.

### The mesh dependence, and `measure`

`alpha_v` in the first cell is a cell AVERAGE over the wall cell thickness, not a
field value at a point - so it changes when the mesh changes, and any criterion
keyed on it inherits that. `measure` chooses how the void is evaluated:
[`NearWallCell`](@ref) accepts the dependence, [`BubblyLayerAverage`](@ref)
removes it by averaging over a physically-set distance.
"""
struct VoidTransition{M,F<:AbstractFloat} <: AbstractTransitionDriver
    measure::M
    alpha_1::F
    alpha_2::F
end
Adapt.@adapt_structure VoidTransition

function VoidTransition(; measure = NearWallCell(), alpha_1 = 0.8, alpha_2 = 0.95)
    0 <= alpha_1 < alpha_2 <= 1 || throw(ArgumentError(
        "need 0 <= alpha_1 < alpha_2 <= 1, got alpha_1 = $alpha_1, alpha_2 = $alpha_2"))
    return VoidTransition(measure, float(alpha_1), float(alpha_2))
end


# =============================================================================
#  How the near-wall void is measured
# =============================================================================

abstract type AbstractVoidMeasure end

"""
    NearWallCell()

Vapour fraction of the wall-adjacent cell, `1 - alpha[cID]`.

Simple and free - the value is already in the kernel. But it is a cell AVERAGE
over the wall cell thickness `delta`, so refining the mesh changes it: it is an
integral divided by `delta`, evaluated over a different slice of a steep profile
each time. A DNB criterion built on it is therefore mesh dependent, and the
threshold that works on one mesh will not transfer.

Use it to explore, or where the near-wall mesh is fixed. Use
[`BubblyLayerAverage`](@ref) for anything that has to be defensible across
meshes.
"""
struct NearWallCell <: AbstractVoidMeasure end
Adapt.@adapt_structure NearWallCell

"""
    BubblyLayerAverage(; cap)

Volume-averaged vapour fraction over a wall-normal layer of thickness

    L = min(D_d, cap)

with `D_d` the local bubble departure diameter.

### Why this is mesh independent

The quantity is an integral over a distance the PHYSICS sets, not one the mesh
sets:

    alpha_bar = (1/L) * integral of alpha_v dy from 0 to L

Refining the mesh then changes only how accurately that integral is evaluated,
so the value CONVERGES rather than drifting. That is the difference between a
discretisation error and a mesh dependence, and it is what makes a threshold
transferable.

`D_d` is the natural length: Weisman & Pei's criterion is on the void in the
BUBBLY LAYER, whose thickness scales with bubble size, and `D_d` is already
computed by the departure model - so the same length that sets `A_b` and `q_e`
sets the averaging layer, with no new constant.

### Why `cap` is required

`D_d` is not guaranteed to stay near-wall. On the LH2 pipe the Fritz value is
1.11 mm against a 3 mm pipe radius - 37% of the radius - so an uncapped layer
would reach past the entire O-grid ring into the core and the criterion would
stop being a near-wall measure at all.

`cap` bounds it. Something like `0.2*R` keeps the layer inside the boundary
region. If the cap binds routinely that is itself worth knowing: a departure
diameter comparable to the channel is a warning that the dispersed-bubbly
assumption underneath RPI, the drift flux and `DispersedBubbles` is strained.
"""
struct BubblyLayerAverage{F<:AbstractFloat} <: AbstractVoidMeasure
    cap::F
end
function BubblyLayerAverage(; cap)
    cap > 0 || throw(ArgumentError(
        "`cap` must be a positive layer thickness [m], got $cap"))
    return BubblyLayerAverage(float(cap))
end
Adapt.@adapt_structure BubblyLayerAverage

"""Layer thickness actually used: the departure diameter, bounded by the cap."""
@inline void_layer_thickness(::NearWallCell, D_d::F) where F = zero(F)
@inline void_layer_thickness(m::BubblyLayerAverage, D_d::F) where F =
    min(D_d, F(m.cap))

@inline _void_by_measure(::NearWallCell, alpha_cell::F, alpha_layer) where F =
    one(F) - alpha_cell
@inline _void_by_measure(::BubblyLayerAverage, alpha_cell, alpha_layer) = alpha_layer

@inline needs_layer_average(::AbstractVoidMeasure) = false
@inline needs_layer_average(::BubblyLayerAverage) = true
@inline needs_layer_average(::Nothing) = false
@inline needs_layer_average(t::VoidTransition) = needs_layer_average(t.measure)
@inline needs_layer_average(::SuperheatTransition) = false


"""
    FilmBoiling(; chf, minimum_film, htc, transition, min_width=0.05)

Post-CHF wall treatment: blends the RPI nucleate partition into a film boiling
flux across the transition region, giving one continuous `q_w(T_w)` from onset
of boiling through DNB to fully established film boiling.

Attach it to [`RPI`](@ref) via its `film_boiling` keyword.

### Keywords
- `chf`          -- [`Zuber`](@ref), [`BubbleCrowding`](@ref) or
                    [`FixedCriticalHeatFlux`](@ref).
- `minimum_film` -- [`Berenson`](@ref), [`HomogeneousNucleation`](@ref),
                    [`FixedMinimumFilmBoiling`](@ref), or a `Tuple` of two, in
                    which case the smaller superheat is used.
- `htc`          -- [`ForcedConvectionFilm`](@ref) (default) or [`Bromley`](@ref).
- `min_width`    -- numerical floor on the width of the transition, as a fraction
                    of the CHF superheat. Purely a guard: if the two correlations
                    return `dT_min <= dT_CHF` the blend interval would be empty
                    or inverted, and this keeps it ordered and non-degenerate.
                    It does not shape the physics anywhere the correlations
                    already give a sensible interval.

### How the blend is formed

Per wall face, once per timestep:

    q_CHF  = critical_heat_flux(chf, state)
    dT_lo  = superheat at which the RPI partition delivers q_CHF
    dT_hi  = max(minimum_film_superheat(minimum_film, state),
                 dT_lo*(1 + min_width))
    h_f    = film_boiling_htc(htc, state, y_plus, u_tau)

then, as a function of wall temperature,

    w      = smoothstep((dT_sup - dT_lo)/(dT_hi - dT_lo))
    q_w    = (1 - w) min(q_c + q_q + q_e, q_CHF) + w h_f max(dT_sup, dT_hi)

`w` is the cubic smoothstep `x^2(3 - 2x)`, so `q_w(T_w)` is continuously
differentiable at both ends of the transition. A discontinuous switch there is
what makes a hard DNB criterion chatter under prescribed flux.

Two details in that expression are not cosmetic. The nucleate branch is CAPPED at
`q_CHF` rather than merely de-weighted, because `N_a ~ dT_sup^n` with a calibrated
`n` of order 20 outruns any linear weight and would spike the curve instead of
turning it over; capping is not an extra tuning constant but the definition of
CHF. And the film term uses `max(dT_sup, dT_hi)` so that it equals `h_f dT_hi` at
the top of the blend and `h_f dT_sup` beyond it, joining the film branch without
a step. See the four-argument [`wall_heat_partition`](@ref).

Note that `dT_lo` comes from inverting the RPI model itself, not from a separate
correlation - `q_RPI(dT)` is monotone, so the superheat matching `q_CHF` is
unique and needs no new empirical input.

### Requires a wall thermal capacity

`q_w(T_w)` DECREASES through the transition, which is the physical boiling curve
and the reason DNB is a jump. Inverting a non-monotone curve at prescribed flux
has up to three roots, so `FilmBoiling` requires `RPI(wall_capacity > 0)`: the
wall energy balance is integrated in time and follows the curve instead of
selecting a root. `RPI` rejects the combination at construction.

### Example
```julia
wall_boiling = RPI(
    patches = (:pipeWall,),
    wall_capacity = 24.21,                    # SS308, 0.5 mm, cryogenic cp
    film_boiling = FilmBoiling(
        chf = Zuber(),
        minimum_film = (Berenson(), HomogeneousNucleation(T_crit = 33.15)),
        htc = ForcedConvectionFilm()),
)
```
"""
struct FilmBoiling{C,M,H,T,F<:AbstractFloat}
    chf::C
    minimum_film::M
    htc::H
    transition::T
    min_width::F
end
Adapt.@adapt_structure FilmBoiling

function FilmBoiling(; chf = nothing, minimum_film, htc = ForcedConvectionFilm(),
                       transition = SuperheatTransition(), min_width = 0.05)
    mf = minimum_film isa Tuple ? _min_of(minimum_film...) : minimum_film
    min_width > 0 || throw(ArgumentError(
        "`min_width` must be positive - it is the floor on the transition width, got $min_width"))

    # `chf` is REQUIRED by the superheat driver, which cannot locate `dT_lo`
    # without it, and UNUSED by the void driver - see the note on the nucleate
    # cap below. Rejecting the missing case here beats discovering it as a
    # silently degenerate blend at run time.
    if transition isa SuperheatTransition && chf === nothing
        throw(ArgumentError(
            "`SuperheatTransition` needs a `chf` closure: it locates the transition by\n" *
            "inverting the RPI partition at the critical heat flux. Supply one, e.g.\n" *
            "`chf = Zuber()` or `chf = FixedCriticalHeatFlux(q = ...)`.\n\n" *
            "`VoidTransition` needs no CHF value at all - departure emerges from the\n" *
            "near-wall vapour fraction - so `chf` may be omitted there."))
    end
    return FilmBoiling(chf, mf, htc, transition, float(min_width))
end

_min_of(a) = a
_min_of(a, b) = MinOfTwo(a, b)
_min_of(a, b, rest...) = MinOfTwo(a, _min_of(b, rest...))

# --- what the transition reads, resolved from the model -----------------------
# Defined here rather than beside the measures because they dispatch on
# `FilmBoiling`, which is not in scope until now.

@inline needs_layer_average(f::FilmBoiling) = needs_layer_average(f.transition)

"""
    wall_void_fraction(film, alpha_cell, alpha_layer) -> alpha_v

Which vapour fraction the transition is driven by. `alpha_cell` is the tracked
(liquid) fraction of the wall cell; `alpha_layer` is the layer average the solver
maintains. Dispatched, so the unused branch compiles out and a superheat-driven
case never touches either.
"""
@inline wall_void_fraction(::Nothing, alpha_cell::F, alpha_layer) where F = zero(F)
@inline wall_void_fraction(f::FilmBoiling, alpha_cell, alpha_layer) =
    _void_from(f.transition, alpha_cell, alpha_layer)

@inline _void_from(::SuperheatTransition, alpha_cell::F, alpha_layer) where F = zero(F)
@inline _void_from(t::VoidTransition, alpha_cell, alpha_layer) =
    _void_by_measure(t.measure, alpha_cell, alpha_layer)

"""
    recorded_void_fraction(film, alpha_cell, alpha_layer) -> alpha_v

What the `alpha_wall` diagnostic should hold.

Differs from [`wall_void_fraction`](@ref) in the one case that matters for
output: a superheat-driven model does not USE a void fraction, so
`wall_void_fraction` returns zero for it - but writing zero into the diagnostic
would be misleading. This returns the wall-cell value there instead, so the field
means "near-wall vapour fraction" in every run regardless of which driver is
active.
"""
@inline recorded_void_fraction(::Nothing, alpha_cell::F, alpha_layer) where F =
    one(F) - alpha_cell
@inline recorded_void_fraction(f::FilmBoiling, alpha_cell, alpha_layer) =
    _record_by(f.transition, alpha_cell, alpha_layer)
@inline _record_by(::SuperheatTransition, alpha_cell::F, alpha_layer) where F =
    one(F) - alpha_cell
@inline _record_by(t::VoidTransition, alpha_cell, alpha_layer) =
    _void_by_measure(t.measure, alpha_cell, alpha_layer)

"""Layer thickness handed to the next averaging pass; zero when unused."""
@inline void_layer_thickness_for(::Nothing, D_d::F) where F = zero(F)
@inline void_layer_thickness_for(f::FilmBoiling, D_d) = _layer_for(f.transition, D_d)
@inline _layer_for(::SuperheatTransition, D_d::F) where F = zero(F)
@inline _layer_for(t::VoidTransition, D_d) = void_layer_thickness(t.measure, D_d)

"""
    FilmClosure(model, h_f, q_chf, dT_lo, dT_hi)

The film boiling blend resolved to numbers for one wall face, so the wall
temperature solve can evaluate it without redoing the CHF inversion at every
bisection step. Built by [`film_closure`](@ref).

`q_chf` is carried because it caps the nucleate branch as well as locating
`dT_lo` - see the discussion in the four-argument
[`wall_heat_partition`](@ref).
"""
struct FilmClosure{M,F<:AbstractFloat}
    model::M
    h_f::F
    q_chf::F
    dT_lo::F
    dT_hi::F
end
Adapt.@adapt_structure FilmClosure

"""
    film_closure(rpi, film, state, h_c, y_plus, u_tau) -> FilmClosure or nothing

Resolve the transition endpoints and film heat transfer coefficient for one wall
face. Called ONCE per face per timestep, outside the wall temperature iteration.

The CHF superheat is found by inverting the nucleate partition with
[`solve_wall_temperature`](@ref), which costs one extra bisection per face.
That is on a boundary-only kernel, so it is negligible beside the field solves -
and it is what keeps the blend a cheap pure function of `T_w` inside the loop.
"""
@inline film_closure(rpi, ::Nothing, s, h_c, y_plus, u_tau) = nothing

@inline film_closure(rpi, film::FilmBoiling, s::BoilingState{F},
                     h_c, y_plus, u_tau) where F =
    _film_closure(film.transition, rpi, film, s, h_c, y_plus, u_tau)

"""
Void-driven closure. NEITHER the CHF value nor the transition superheats are
used, so none of them is computed.

### Why there is no nucleate cap here

The cap exists for the superheat driver, where `w` and `dT_sup` are the SAME
variable: as the wall heats through the transition, `(1 - w)*q_RPI(dT_sup)` is
evaluated at ever larger superheat, and with `q_evap ~ dT_sup^n` at n ~ 21 that
outruns the linear weight - measured at 10^7 kW/m^2 mid-transition on the LH2
pipe. Capping at `q_CHF` is what turns that spike back into a turnover.

A void-driven blend has no such coupling. `w` is a function of `alpha_v`, so at
any fixed void the wall flux

    Q(T_w) = (1 - w)*q_RPI(T_w) + w*h_f*dT_sup

is MONOTONE INCREASING in `T_w`; the wall temperature solve has a unique root and
cannot run away. Once `alpha_v` passes `alpha_2` the `(1 - w)` factor removes the
nucleate term outright, so there is nothing left for a cap to bound.

Dropping it removes the last dependency on a critical heat flux value, which is
the entire point of this driver, and with it a real failure mode: `Zuber`
evaluated before the vapour density is initialised returns zero, and a zero cap
annihilates the whole partition at every wall temperature.
"""
@inline function _film_closure(::VoidTransition, rpi, film, s::BoilingState{F},
                               h_c, y_plus, u_tau) where F
    h_f = film_boiling_htc(film.htc, s, y_plus, u_tau)
    # `Inf` disables the nucleate cap through the guard in `_nucleate_cap`; the
    # superheat endpoints are unused by this driver and are marked unusable so
    # nothing can read them by accident.
    return FilmClosure(film, h_f, F(Inf), F(Inf), F(Inf))
end

@inline function _film_closure(::SuperheatTransition, rpi, film, s::BoilingState{F},
                               h_c, y_plus, u_tau) where F

    q_chf = critical_heat_flux(film.chf, s, rpi, h_c)

    # Superheat matching CHF, from the nucleate model's own inversion.
    T_chf, _ = solve_wall_temperature(rpi, s, q_chf, h_c)
    dT_lo = max(T_chf - s.T_sat, zero(F))

    dT_hi_raw = minimum_film_superheat(film.minimum_film, s)
    # Order and separate the interval. Both correlations are approximate and
    # nothing couples them, so `dT_hi <= dT_lo` is a perfectly reachable outcome
    # - especially for a cryogen, where the homogeneous nucleation bound sits
    # only a few kelvin above saturation. Clamping here turns that into a narrow
    # but well-posed transition, which is the physically right reading of it
    # (an abrupt DNB), instead of an inverted or zero-width blend.
    dT_hi = max(dT_hi_raw, dT_lo*(one(F) + F(film.min_width)),
                dT_lo + eps(F))

    h_f = film_boiling_htc(film.htc, s, y_plus, u_tau)

    return FilmClosure(film, h_f, q_chf, dT_lo, dT_hi)
end

# `BubbleCrowding` needs the RPI sub-models, so `critical_heat_flux` takes the
# model and h_c as well. The property-only closures ignore both, which is the
# point of them.
@inline critical_heat_flux(m::AbstractCriticalHeatFlux, s, rpi, h_c) =
    critical_heat_flux(m, s)

@inline function critical_heat_flux(
    model::BubbleCrowding, s::BoilingState{F}, rpi, h_c) where F
    # Find the superheat at which A_b reaches A_crit, then report the partition
    # total there. `A_b` is monotone in superheat through `N_a`, so bisection on
    # it is as well posed as the flux inversion is.
    lo = zero(F)
    hi = one(F)                        # 1 K seed, doubled below until covered
    for _ in 1:12
        p = wall_heat_partition(rpi, _at_wall_temperature(s, s.T_sat + hi), h_c)
        hi = p.A_b < F(model.A_crit) ? 2*hi : hi
    end
    for _ in 1:rpi.n_iterations
        mid = (lo + hi)/2
        p = wall_heat_partition(rpi, _at_wall_temperature(s, s.T_sat + mid), h_c)
        if p.A_b < F(model.A_crit)
            lo = mid
        else
            hi = mid
        end
    end
    dT = (lo + hi)/2
    p = wall_heat_partition(rpi, _at_wall_temperature(s, s.T_sat + dT), h_c)
    return p.q_c + p.q_q + p.q_e
end

"""
    film_boiling_fraction(fc, dT_sup, alpha_v) -> w in [0, 1]

Weight of the film boiling closure. Zero on the nucleate branch, one once the
film is established, cubic smoothstep between.

WHAT drives it is set by the model's `transition`:
[`SuperheatTransition`](@ref) uses `dT_sup`, [`VoidTransition`](@ref) uses the
near-wall vapour fraction `alpha_v`. Both arguments are always passed so the
kernel keeps one shape; the unused one compiles out.
"""
@inline film_boiling_fraction(::Nothing, dT_sup::F, alpha_v) where F = zero(F)

@inline film_boiling_fraction(fc::FilmClosure, dT_sup, alpha_v) =
    _transition_weight(fc.model.transition, fc, dT_sup, alpha_v)

@inline _smoothstep(x::F) where F =
    (y = clamp(x, zero(F), one(F)); y*y*(3 - 2*y))   # C1 at both ends

# Superheat used by the FILM term. The `max(dT_sup, dT_min)` exists only to join
# the superheat blend continuously to the film branch at the top of its interval;
# a void-driven blend has no such join, and `dT_hi` is not even computed for it.
# See `_film_flux` in `2_wall_boiling_models.jl`.
@inline _film_dT(::SuperheatTransition, dT_sup::F, dT_hi) where F =
    isfinite(dT_hi) ? max(dT_sup, dT_hi) : max(dT_sup, zero(F))
@inline _film_dT(::VoidTransition, dT_sup::F, dT_hi) where F = max(dT_sup, zero(F))

@inline function _transition_weight(
    ::SuperheatTransition, fc, dT_sup::F, alpha_v) where F
    lo, hi = F(fc.dT_lo), F(fc.dT_hi)
    # An unusable interval means the CHF closure could not be evaluated - a
    # property not yet initialised, a state outside the correlation. Stay on the
    # nucleate branch rather than producing `Inf - Inf`.
    (isfinite(lo) && isfinite(hi) && hi > lo) || return zero(F)
    return _smoothstep((dT_sup - lo)/(hi - lo))
end

@inline _transition_weight(t::VoidTransition, fc, dT_sup, alpha_v::F) where F =
    _smoothstep((alpha_v - F(t.alpha_1))/(F(t.alpha_2) - F(t.alpha_1)))
