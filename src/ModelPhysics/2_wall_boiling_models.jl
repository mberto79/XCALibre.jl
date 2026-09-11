export AbstractWallBoilingModel, RPI, wall_boiling_active
export AbstractBubblyLayer, WallCellLayer, DiameterLayer, YPlusLayer, WallSurface
export PlayHysteresis, play_update
export MassBalanceLayer, mass_balance_void
export bubbly_layer_void, bubbly_layer_thickness, dryout_snap_void
export AbstractNucleationSiteDensity, LemmertChawla, HibikiIshii, Kirichenco
export AbstractDepartureDiameter, TolubinskyKostanchuk, KocamustafaogullariIshii, Du
export AbstractDepartureFrequency, Cole, BaldGrowth
export AbstractInfluenceArea, DelValleKenning, ConstantInfluenceArea
export BoilingState
export nucleation_site_density, bubble_departure_diameter
export bubble_departure_frequency, bubble_influence_fraction
export wall_heat_partition, solve_wall_temperature, solve_wall_temperature_transient, single_phase_htc
export wall_boiling_patches, wall_boiling_liquid_factor

# =============================================================================
#  Wall nucleate boiling - RPI heat flux partitioning
# =============================================================================
#
#  The RPI model (Kurul & Podowski, 1990) splits the heat leaving a boiling wall
#  into three parallel paths,
#
#      q_w = q_c + q_q + q_e
#
#  q_c  single-phase convection over the wall area NOT influenced by bubbles,
#  q_q  quenching - transient conduction into the cold liquid that rushes in to
#       fill the space a departing bubble leaves behind,
#  q_e  evaporation - the latent heat carried away by the bubbles themselves.
#
#  Only q_e generates vapour. The other two heat the liquid, which is why the
#  partition matters: for the same wall flux, a split that puts more into q_e
#  produces more vapour and a lower wall superheat.
#
#  Each of the four empirical closures below is an independent, swappable piece.
#  Adding one means declaring a struct under the relevant supertype and giving
#  it a method - nothing else in the solver changes.
# =============================================================================

abstract type AbstractWallBoilingModel end

abstract type AbstractNucleationSiteDensity end
abstract type AbstractDepartureDiameter end
abstract type AbstractDepartureFrequency end
abstract type AbstractInfluenceArea end

"""
    BoilingState

Local thermodynamic and transport state at one wall face, assembled by the
solver and handed to every sub-model.

A single state object is passed to all four closures - rather than each taking
the arguments it happens to need - so that a new correlation can use any
quantity here without changing a signature anywhere else.

### Fields
- `T_w`, `T_l`, `T_sat` -- wall, near-wall liquid and saturation temperature [K].
- `dT_sup`  -- wall superheat, `T_w - T_sat` [K]. Negative means no boiling.
- `dT_sub`  -- liquid subcooling, `T_sat - T_l` [K]. Negative when the bulk is
               itself superheated.
- `rho_l`, `rho_v` -- liquid and vapour density [kg/m^3].
- `cp_l`, `k_l`, `mu_l` -- liquid heat capacity, conductivity and viscosity.
- `sigma`   -- surface tension [N/m].
- `h_fg`    -- latent heat of vaporisation [J/kg].
- `g`       -- gravitational acceleration magnitude [m/s^2].
"""
struct BoilingState{F<:AbstractFloat}
    T_w::F
    T_l::F
    T_sat::F
    dT_sup::F
    dT_sub::F
    rho_l::F
    rho_v::F
    cp_l::F
    k_l::F
    mu_l::F
    sigma::F
    h_fg::F
    g::F
    cp_v::F
    k_v::F
    mu_v::F
    # LIQUID volume fraction of the wall cell. Defaults to 1 (fully wetted), which
    # reproduces the classical Kurul-Podowski partition exactly. Only the
    # STAR-CCM+ style :mmp partition reads it - see wall_heat_partition.
    alpha_l::F
end
Adapt.@adapt_structure BoilingState

"""
    BoilingState(; T_w, T_l, T_sat, rho_l, rho_v, cp_l, k_l, mu_l, sigma, h_fg, g,
                   cp_v=0, k_v=0, mu_v=0)

Convenience constructor deriving `dT_sup` and `dT_sub` from the temperatures.

The vapour transport properties default to zero because nucleate boiling never
uses them - RPI only needs `rho_v`, through `q_e`. They are required by the film
boiling models, which transport heat *through* the
vapour rather than into it, and those models return zero if they are left unset
rather than silently using liquid values.
"""
function BoilingState(; T_w, T_l, T_sat, rho_l, rho_v, cp_l, k_l, mu_l, sigma, h_fg, g,
                        cp_v = 0, k_v = 0, mu_v = 0, alpha_l = 1)
    F = promote_type(typeof(float(T_w)), typeof(float(rho_l)))
    return BoilingState{F}(
        F(T_w), F(T_l), F(T_sat), F(T_w - T_sat), F(T_sat - T_l),
        F(rho_l), F(rho_v), F(cp_l), F(k_l), F(mu_l), F(sigma), F(h_fg), F(g),
        F(cp_v), F(k_v), F(mu_v), F(alpha_l))
end

"""Same state at a different wall temperature. Used by the wall-temperature solve."""
@inline _at_wall_temperature(s::BoilingState{F}, T_w) where F = BoilingState{F}(
    T_w, s.T_l, s.T_sat, T_w - s.T_sat, s.dT_sub,
    s.rho_l, s.rho_v, s.cp_l, s.k_l, s.mu_l, s.sigma, s.h_fg, s.g,
    s.cp_v, s.k_v, s.mu_v, s.alpha_l)


# =============================================================================
#  Bubbly layer thickness, for the dryout criterion
# =============================================================================

"""
    AbstractBubblyLayer

How the bubbly layer thickness `delta` is set for the wall dryout criterion.

STAR-CCM+ offers three (User Guide, Wall Dryout): the wall cell width, a fixed
number of bubble departure diameters, or a fixed y+ in the VAPOUR turbulence
scales. All three are here.

The layer matters because `K_dry` should respond to the vapour fraction over the
bubbly layer, `alpha_delta`, not to whatever value happens to sit at the first
cell centre - which is a mesh artefact. See [`bubbly_layer_void`](@ref).
"""
abstract type AbstractBubblyLayer end

"""
    WallCellLayer()

`delta` = the wall cell thickness. Cheapest, and the honest choice when the mesh
is already sized on the bubble scale - but it makes the criterion mesh dependent
by construction, which is the thing the layer average exists to avoid.
"""
struct WallCellLayer <: AbstractBubblyLayer end
Adapt.@adapt_structure WallCellLayer

"""
    DiameterLayer(; n = 1.0)

`delta = n*D_d`, a fixed number of bubble departure diameters. Ties the layer to
the physical scale that sets it, and is the option to prefer when `D_d` is
trustworthy.
"""
struct DiameterLayer{F<:AbstractFloat} <: AbstractBubblyLayer
    n::F
end
DiameterLayer(; n = 1.0) = DiameterLayer(float(n))
Adapt.@adapt_structure DiameterLayer

"""
    YPlusLayer(; y_plus = 250.0)

`delta = y_plus * nu_v/u_tau`, a fixed y+ in the VAPOUR turbulence scales.
Independent of both the mesh and the departure diameter, which is what
recommends it when neither is reliable.
"""
struct YPlusLayer{F<:AbstractFloat} <: AbstractBubblyLayer
    y_plus::F
end
YPlusLayer(; y_plus = 250.0) = YPlusLayer(float(y_plus))
Adapt.@adapt_structure YPlusLayer

"""
    WallSurface()

Evaluate the void AT THE WALL (`y = 0`) rather than averaged over a layer:

    alpha_delta = alpha(y_c) - alpha'(y_c)*y_c

Not one of STAR-CCM+'s three, and it is not a layer average at all - it is the
zero-thickness limit of the same expansion. Three reasons it belongs here:

  * `K_dry` is defined as the fraction of WALL AREA not wetted by liquid. That is
    a property of the surface, so the void that decides it arguably belongs at
    the surface rather than 54 um out in the flow.

  * It is the SHORTEST extrapolation on offer. The one-term Taylor expansion is
    only credible over a short distance, and `y_c` (27.9 um on the LH2 pipe) is
    less than `DiameterLayer(n=1)`'s +25.8 um offset and far less than
    `DiameterLayer(n=5.5)`'s +267 um.

  * Its SIGN depends on the profile, which is the point. STAR's guidance to use
    5.5 departure diameters assumes the void RISES away from the wall to a bubbly
    layer peak, so a thicker layer captures more of it. Measured on the LH2 pipe
    the profile does the opposite - 0.626 at 0-56 um decaying monotonically to
    0.454 at 600-900 um, `alpha' = -894 /m` - so a thicker layer averages in more
    BULK and dilutes the criterion. At n = 5.5 the increment is -0.239, which
    puts `K_dry` at zero for every flux in the ladder. `WallSurface()` moves the
    other way, +0.025.

Measured on the LH2 pipe at 1.2e5: `alpha_delta` 0.696 -> ~0.744, `K_dry`
0.48 -> ~0.66 on a 0.5-0.9 ramp. A real gain, but note it is NOT on its own
enough to reach `K_dry = 1`, which is what a wall temperature excursion needs
when `N_a ~ dT_sup^9.945`.
"""
struct WallSurface <: AbstractBubblyLayer end
Adapt.@adapt_structure WallSurface

"""
    MassBalanceLayer(; c_vp = 0.25, inner = 5, relax = 1.0)

Bubbly-layer void from a MASS BALANCE rather than from a gradient extrapolation:

    alpha_bl = alpha_cell + q_E / (rho_v * h_fg * v'),     v' = c_vp*sqrt(k)

The vapour generated at the wall, expressed as a velocity `q_E/(rho_v h_fg)`, is
balanced against the turbulent fluctuating velocity `v'` that pushes it out of
the bubbly layer into the core. The ratio is the void the sub-grid layer must
carry above the resolved cell value.

### Why this and not a layer average

Every finite-thickness layer, and the `WallSurface` limit, reads the RESOLVED
void field. On a wall-function mesh that field cannot show a blanket: with a
55.8 um first cell, a dry film of 16-23 um averages to a cell value of only
~0.64, and reaching 0.9 would need the film to fill 80% of the cell. So a
threshold on the resolved void has to sit within ~0.02 of a peak that is not
predictable in advance - measured on the LH2 pipe, a threshold set at 0.645 from
a projected peak of 0.662 never fired, because the peak was actually 0.624.

The mass balance does not have that problem. It adds a SUB-GRID excess from a
flux balance, so the mesh does not have to resolve the film. On the same LH2 data
it gives 0.806 at the measured CHF of 6.4e4 and 0.849 at 7e4 - a clean crossing
of Weisman & Pei's geometric limit of 0.82, with nothing fitted.

### The inner iteration

`q_E` is produced by the partition that `alpha_bl` feeds (`alpha_bl -> a_l ->
h_c` and `K_dry` -> solve -> `q_E`), so the two are CIRCULAR. This is resolved by
a fixed-point iteration inside the wall solve rather than by lagging `q_E` a
step: lagging `alpha_delta` is a known failure mode on this case, having driven
the `alpha -> h_c -> T_wall -> evaporation` loop to NaN in 60 steps.

The map is contracting in the normal case - raising `alpha_bl` raises `K_dry`,
which cuts `q_E`, which lowers `alpha_bl` - so a handful of iterations suffice.
`relax` under-relaxes it for the case where the wall is near full dryout, where
`q_e ~ dT_sup^n` makes the response very stiff.

`inner` is a FIXED count so the kernel stays branch-free.
"""
struct MassBalanceLayer{F<:AbstractFloat} <: AbstractBubblyLayer
    c_vp::F
    inner::Int
    relax::F
    alpha_min_bl::F
end
function MassBalanceLayer(; c_vp = 0.25, inner = 5, relax = 1.0, alpha_min_bl = 0.01)
    c_vp > 0 || throw(ArgumentError("`c_vp` must be positive, got $c_vp"))
    inner >= 1 || throw(ArgumentError("`inner` must be at least 1, got $inner"))
    0 < relax <= 1 || throw(ArgumentError("`relax` must be in (0, 1], got $relax"))
    0 <= alpha_min_bl < 1 || throw(ArgumentError(
        "`alpha_min_bl` must be in [0, 1), got $alpha_min_bl"))
    return MassBalanceLayer(float(c_vp), inner, float(relax), float(alpha_min_bl))
end
Adapt.@adapt_structure MassBalanceLayer

@inline bubbly_layer_thickness(::Nothing, y_c::F, D_d, nu_v, u_tau) where F = zero(F)
@inline bubbly_layer_thickness(::WallCellLayer, y_c::F, D_d, nu_v, u_tau) where F =
    2*y_c                                   # cell centre sits at half the width
@inline bubbly_layer_thickness(m::DiameterLayer, y_c::F, D_d, nu_v, u_tau) where F =
    F(m.n)*D_d
@inline bubbly_layer_thickness(m::YPlusLayer, y_c::F, D_d, nu_v, u_tau) where F =
    F(m.y_plus)*nu_v/max(u_tau, eps(F))
# `WallSurface` has no layer: the evaluation point is y = 0, not delta/2. The
# thickness is reported as zero and `bubbly_layer_void` dispatches on the TYPE
# rather than reading it - see the note there about why zero cannot simply be
# passed through the generic path.
@inline bubbly_layer_thickness(::WallSurface, y_c::F, D_d, nu_v, u_tau) where F = zero(F)
# `MassBalanceLayer` has no geometric layer at all - the sub-grid excess comes
# from a flux balance, not from a thickness. Reported as zero; the kernel
# dispatches on the type.
@inline bubbly_layer_thickness(::MassBalanceLayer, y_c::F, D_d, nu_v, u_tau) where F = zero(F)

"""
    bubbly_layer_void(layer, alpha_v_cell, dadn, y_c, D_d, nu_v, u_tau) -> alpha_delta

Vapour fraction averaged over the bubbly layer, by the one-term expansion of
STAR-CCM+ User Guide Eqn (2112):

    alpha_delta = (1/delta) * int_0^delta [alpha(y_c) + alpha'(y_c)*(y - y_c)] dy
                = alpha(y_c) + alpha'(y_c)*(delta/2 - y_c)

`dadn` is the WALL-NORMAL derivative of the vapour fraction at the cell centre,
i.e. `grad(alpha_v) . n` with `n` pointing INTO the fluid.

### Why an expansion rather than a cell average

Averaging the cells that fall inside the layer fails whenever the layer is
THINNER than the first cell - no cell qualifies, and the average is undefined.
That is not hypothetical: with `KocamustafaogullariIshii` at the measured 4 deg
contact angle, `D_d = 12.5 um` against a first cell centre of 27.9 um, so a
stencil-based layer average silently returned zero and the criterion could never
fire.

The expansion has no such failure mode. When `delta/2 < y_c` it extrapolates
INWARD, toward the wall, which is exactly the right thing to do when the layer is
finer than the mesh - and it needs no cells inside the layer at all.

The result is clamped to [0,1]: it is a linear extrapolation, so nothing stops it
overshooting on a steep profile.
"""
@inline function bubbly_layer_void(layer, alpha_v_cell::F, dadn, y_c, D_d,
                                   nu_v, u_tau) where F
    delta = bubbly_layer_thickness(layer, y_c, D_d, nu_v, u_tau)
    delta <= zero(F) && return clamp(alpha_v_cell, zero(F), one(F))
    return clamp(alpha_v_cell + dadn*(F(0.5)*delta - y_c), zero(F), one(F))
end

# `WallSurface` needs its own method rather than a zero `delta`, because the
# generic path treats `delta <= 0` as "no layer information" and returns the raw
# cell value. Here zero thickness is MEANINGFUL - it is the wall itself - so the
# expansion is evaluated at `y = 0`:
#
#     alpha(0) = alpha(y_c) + alpha'(y_c)*(0 - y_c)
#
# Note the sign: with `alpha'` NEGATIVE (void decaying away from the wall, which
# is what the LH2 pipe does) this INCREASES the result, whereas every
# finite-thickness layer decreases it.
@inline bubbly_layer_void(::WallSurface, alpha_v_cell::F, dadn, y_c, D_d,
                          nu_v, u_tau) where F =
    clamp(alpha_v_cell - dadn*y_c, zero(F), one(F))

# `MassBalanceLayer` needs `q_E`, which this signature does not carry. The
# pre-pass therefore seeds it with the raw cell value and the inner iteration in
# the wall kernel does the real work - see `mass_balance_void`.
@inline bubbly_layer_void(::MassBalanceLayer, alpha_v_cell::F, dadn, y_c, D_d,
                          nu_v, u_tau) where F =
    clamp(alpha_v_cell, zero(F), one(F))

"""
    mass_balance_void(m, alpha_v_cell, q_E, rho_v, h_fg, k_turb)

`alpha_cell + q_E/(rho_v*h_fg*c_vp*sqrt(k))`, clamped to [0,1].

`k_turb` is the turbulent kinetic energy in the wall cell. A floor is applied to
`v'` so a laminar or unseeded cell cannot divide by zero - there the balance is
meaningless and the cell value is the right answer.
"""
@inline function mass_balance_void(m::MassBalanceLayer, alpha_v_cell::F, q_E,
                                   rho_v, h_fg, k_turb) where F
    # NO BUBBLY LAYER, NO BALANCE. The expression balances vapour generated at
    # the wall against turbulent transport OUT OF an existing bubbly layer, so it
    # is meaningless before one exists.
    #
    # Applying it anyway latches the model dry and diverges. At cold start
    # `alpha_v_cell ~ 0` and `k` is small, so the increment is enormous and
    # `alpha_bl` clamps to 1; that sets `K_dry = 1`, which zeroes `q_E`, so no
    # vapour is ever generated and `alpha_v_cell` stays 0 - while the wall must
    # carry the full flux through vapour-property convection alone. Measured: the
    # velocity field reached Courant 9.7e132 by step 88.
    #
    # `alpha_min_bl` is the same 0.01 the alpha-Courant gate uses for "this cell
    # contains an interface".
    alpha_v_cell > F(m.alpha_min_bl) || return clamp(alpha_v_cell, zero(F), one(F))
    vp = F(m.c_vp)*sqrt(max(k_turb, zero(F)))
    den = rho_v*h_fg*vp
    den > eps(F) || return clamp(alpha_v_cell, zero(F), one(F))
    return clamp(alpha_v_cell + max(q_E, zero(F))/den, zero(F), one(F))
end


# =============================================================================
#  Nucleation site density
# =============================================================================

"""
    nucleation_site_density(model, state) -> sites/m^2

Active nucleation site density on the heated wall. Zero at or below saturation.
"""
function nucleation_site_density end

"""
    LemmertChawla(; m=210.0, n=1.805)

Lemmert & Chawla (1977) nucleation site density,

    N_a = (m * dT_sup)^n            [1/m^2]

with `m = 210` [1/(m K)] and `n = 1.805` as originally fitted, which is the
form used in essentially every RPI implementation (and the default in CFX and
STAR-CCM+).

The exponent is what makes wall boiling stiff: `N_a` grows almost as the square
of the superheat, so the evaporative flux is very sensitive to `T_w`. That is
physical, not a numerical artefact, and it is why the wall temperature is
obtained by a bracketed solve rather than by iterating the partition directly.
"""
struct LemmertChawla{F<:AbstractFloat} <: AbstractNucleationSiteDensity
    m::F
    n::F
end
LemmertChawla(; m=210.0, n=1.805) = LemmertChawla(float(m), float(n))
Adapt.@adapt_structure LemmertChawla

@inline function nucleation_site_density(model::LemmertChawla, s::BoilingState{F}) where F
    s.dT_sup <= zero(F) && return zero(F)
    return (F(model.m)*s.dT_sup)^F(model.n)
end

"""
    HibikiIshii(; N_bar=4.72e5, lambda=2.5e-6, mu_c=0.722, theta=0.722, N_max=1.0e12)

Hibiki & Ishii (2003) mechanistic site density, provided as a second option so
the sensitivity of a result to this closure can be tested.

    N_a = N_bar * (1 - exp(-theta^2/(8 mu_c^2))) * (exp(f(rho+) * lambda/R_c) - 1)

with the critical cavity radius from Clausius-Clapeyron,

    R_c    = 2 sigma T_sat / (rho_v h_fg dT_sup)
    rho+   = log10((rho_l - rho_v)/rho_v)
    f(rho+) = -0.01064 + 0.48246 rho+ - 0.22712 rho+^2 + 0.05468 rho+^3

`N_bar` is the site density scale [1/m^2], `lambda` the cavity length scale [m],
and `theta`, `mu_c` the contact angle and cavity distribution width, both in
radians (they appear only as the ratio `theta/mu_c`, so the units cancel).

Unlike [`LemmertChawla`](@ref) this responds to pressure through `sigma`,
`rho_v` and `T_sat`, which is the reason to reach for it near the critical
point.

!!! warning "Outside its validation range for liquid hydrogen"
    The correlation was fitted to water and nitrogen. For hydrogen at the
    pressures of the Tatsumoto pipe cases the surface tension is small enough
    that `R_c` falls to tens of nanometres at 1 K of superheat, `lambda/R_c`
    reaches O(100), and the exponential predicts a site density beyond any
    physical wall. `N_max` caps it for that reason.

    In practice the cap binds almost immediately for LH2, which makes this model
    effectively constant there — so it is **not** a drop-in substitute for
    `LemmertChawla` in these cases. It is retained because it is the right
    starting point for fluids inside its fitted range, and because having a
    second site-density model is what demonstrates the interface.
"""
struct HibikiIshii{F<:AbstractFloat} <: AbstractNucleationSiteDensity
    N_bar::F
    lambda::F
    mu_c::F
    theta::F
    N_max::F
end
HibikiIshii(; N_bar=4.72e5, lambda=2.5e-6, mu_c=0.722, theta=0.722, N_max=1.0e12) =
    HibikiIshii(float(N_bar), float(lambda), float(mu_c), float(theta), float(N_max))
Adapt.@adapt_structure HibikiIshii

@inline function nucleation_site_density(model::HibikiIshii, s::BoilingState{F}) where F
    s.dT_sup <= zero(F) && return zero(F)
    (s.sigma <= zero(F) || s.rho_v <= zero(F)) && return zero(F)

    # Critical cavity radius from Clausius-Clapeyron.
    R_c = 2*s.sigma*s.T_sat/(s.rho_v*s.h_fg*s.dT_sup)
    R_c <= zero(F) && return zero(F)

    contact = one(F) - exp(-model.theta^2/(8*model.mu_c^2))

    # Density-ratio function of the original correlation.
    drho = s.rho_l - s.rho_v
    drho <= zero(F) && return zero(F)
    rp = log10(drho/s.rho_v)
    fr = F(-0.01064) + F(0.48246)*rp - F(0.22712)*rp^2 + F(0.05468)*rp^3

    # Cap the EXPONENT only to keep `exp` finite, then cap the site density
    # itself. Capping N_a rather than the argument is what keeps the model
    # strictly increasing in superheat right up to the limit, instead of going
    # flat as soon as the exponential is clipped.
    arg = min(max(fr, zero(F))*model.lambda/R_c, F(80))
    N_a = model.N_bar*contact*(exp(arg) - one(F))
    return min(N_a, model.N_max)
end


"""
    Kirichenco(; N=1.0e-7, m=2.0, N_max=1.0e12)

Kirichenko, Dolgoy & Levchenko (1976) nucleation site density for CRYOGENIC
fluids,

    N_a = N * ( rho_v * h_fg * dT_sup / (sigma * T_sat) )^m

with the bracketed group carrying units of 1/m, so `m = 2` is dimensionally
consistent.

### Why it belongs here

Every other site-density correlation in this file was fitted to water
([`LemmertChawla`](@ref), [`HibikiIshii`](@ref)) or to room-temperature
organics. This one was fitted to cryogens, and Kuang et al. (2021), Int. J.
Hydrogen Energy 46:19617, adopt it specifically for liquid hydrogen after
finding water correlations unusable - their contrast model built on
Kocamustafaogullari-Ishii site density gave a mean absolute error of 52.5%
against 8.94% for this one.

### Coefficients

The original gives two branches on reduced pressure:

    p/p_cr >= 0.04 :  N = 1.0e-7,   m = 2      (the defaults)
    p/p_cr <  0.04 :  N = 6.25e-6,  m = 3

For hydrogen `p_cr = 1.2964 MPa`, so a 0.4 MPa case is at `p/p_cr = 0.31` and
takes the default branch. Pass `N=6.25e-6, m=3.0` below 52 kPa.

### What to expect relative to `LemmertChawla`

The superheat exponent is **2**, against the 1.805 of the unmodified
Lemmert-Chawla and the much steeper exponents that fitting `LemmertChawla` to a
cryogen tends to produce. So `N_a` responds far more gently to `dT_sup`, which
removes the extreme sensitivity that a near-tenth-power law creates in the
wall-temperature solve.

Sample value for LH2 at 0.4 MPa (`rho_v = 4.84`, `h_fg = 393.5 kJ/kg`,
`sigma = 9.49e-4 N/m`, `T_sat = 26.08 K`) at `dT_sup = 1.75 K`: the group is
1.35e8 /m and `N_a ~ 1.8e9 /m^2`. That is high by water standards and is meant
to be - hydrogen nucleates at superheats of order 0.1 K, and CHF is reached
near 3 K.
"""
struct Kirichenco{F<:AbstractFloat} <: AbstractNucleationSiteDensity
    N::F
    m::F
    N_max::F
end
Kirichenco(; N=1.0e-7, m=2.0, N_max=1.0e12) = Kirichenco(float(N), float(m), float(N_max))
Adapt.@adapt_structure Kirichenco

@inline function nucleation_site_density(model::Kirichenco, s::BoilingState{F}) where F
    s.dT_sup <= zero(F) && return zero(F)
    (s.sigma <= zero(F) || s.T_sat <= zero(F) || s.rho_v <= zero(F)) && return zero(F)
    group = s.rho_v*s.h_fg*s.dT_sup/(s.sigma*s.T_sat)
    group <= zero(F) && return zero(F)
    return min(F(model.N)*group^F(model.m), F(model.N_max))
end


# =============================================================================
#  Bubble departure diameter
# =============================================================================

"""
    bubble_departure_diameter(model, state) -> m

Diameter at which a bubble detaches from the wall.
"""
function bubble_departure_diameter end

"""
    TolubinskyKostanchuk(; d_ref=0.6e-3, dT_ref=45.0, d_max=1.4e-3, d_min=1.0e-6)

Tolubinsky & Kostanchuk (1970),

    D_d = min(d_max, d_ref * exp(-dT_sub/dT_ref))

the standard default in RPI implementations. Subcooling shrinks the departing
bubble because the cold liquid condenses its cap while it is still attached.

The original coefficients were fitted to water. For cryogens they are not
established, so `d_ref` and `dT_ref` are exposed as the primary tuning handles
of the wall boiling model - a point worth stating explicitly when reporting
results rather than treating the defaults as physics.
"""
struct TolubinskyKostanchuk{F<:AbstractFloat} <: AbstractDepartureDiameter
    d_ref::F
    dT_ref::F
    d_max::F
    d_min::F
end
TolubinskyKostanchuk(; d_ref=0.6e-3, dT_ref=45.0, d_max=1.4e-3, d_min=1.0e-6) =
    TolubinskyKostanchuk(float(d_ref), float(dT_ref), float(d_max), float(d_min))
Adapt.@adapt_structure TolubinskyKostanchuk

@inline function bubble_departure_diameter(model::TolubinskyKostanchuk, s::BoilingState{F}) where F
    d = F(model.d_ref)*exp(-max(s.dT_sub, zero(F))/F(model.dT_ref))
    return clamp(d, F(model.d_min), F(model.d_max))
end

"""
    KocamustafaogullariIshii(; theta_deg=41.37, d_min=1.0e-8, d_max=1.4e-2)

Kocamustafaogullari & Ishii (1983) force-balance departure diameter: the Fritz
diameter scaled by a density-ratio correction,

    D_d,Fritz = 0.0208 * theta_deg * sqrt(sigma/(g*drho))
    D_d       = 0.0012 * (drho/rho_v)^0.9 * D_d,Fritz

with `drho = rho_l - rho_v`.

`theta_deg` is the contact angle in **degrees**, which is what Fritz's 0.0208
coefficient is written for. Passing radians instead - 0.722 rad is the same
angle as the 41.37 degree default - silently shrinks the result by a factor of
57, which for hydrogen is enough to drive it straight into `d_min`; hence the
explicit name.

Scales with the capillary length rather than with subcooling, so unlike
[`TolubinskyKostanchuk`](@ref) it responds to pressure. That is what recommends
it for hydrogen near its critical point, where the surface tension falls towards
zero and departure diameters collapse with it.
"""
struct KocamustafaogullariIshii{F<:AbstractFloat} <: AbstractDepartureDiameter
    theta_deg::F
    d_min::F
    d_max::F
end
KocamustafaogullariIshii(; theta_deg=41.37, d_min=1.0e-8, d_max=1.4e-2) =
    KocamustafaogullariIshii(float(theta_deg), float(d_min), float(d_max))
Adapt.@adapt_structure KocamustafaogullariIshii

@inline function bubble_departure_diameter(model::KocamustafaogullariIshii, s::BoilingState{F}) where F
    drho = s.rho_l - s.rho_v
    (drho <= zero(F) || s.sigma <= zero(F) || s.rho_v <= zero(F)) && return model.d_min
    d_fritz = F(0.0208)*model.theta_deg*sqrt(s.sigma/(s.g*drho))
    d = F(0.0012)*(drho/s.rho_v)^F(0.9)*d_fritz
    return clamp(d, model.d_min, model.d_max)
end


"""
    Du(; G, C=1.5705e7, d_min=1.0e-8, d_max=1.4e-2)

Du, Zhao & Bo (2018) departure diameter, as adopted for hydrogen by Kuang et al.
(2021),

    d_b/L_c = C * rho*^-0.319 * Ja^0.123 * Pr^-1.939 * Re_b^-0.751

    L_c = rho_l*nu_l^2/sigma      rho* = rho_v/rho_l
    Ja  = cp_l*dT_sup/h_fg        Re_b = G*d_b/mu_l

`G` is the MASS FLUX [kg/m^2/s] and has no default - `BoilingState` does not
carry one, and the correlation cannot be evaluated without it. For a fixed-inlet
case use `rho_l*U_inlet`.

### Why it is worth having

Every other departure model here reduces to a buoyancy-surface-tension balance
through `sqrt(sigma/(g*(rho_l - rho_v)))`, which is the POOL boiling mechanism.
In flow boiling the shear-induced lift detaches bubbles earlier, so departure
diameters are smaller and fall with flow rate. This correlation carries that
dependence explicitly through `Re_b`, which is why Kuang et al. selected it over
Unal and over Kocamustafaogullari-Ishii.

### Implicit form

`Re_b` contains `d_b`, so the correlation is implicit. With the exponent -0.751
it inverts in closed form and no iteration is needed:

    d_b^1.751 = L_c * C * rho*^-0.319 * Ja^0.123 * Pr^-1.939 * (G/mu_l)^-0.751

### On the coefficient

The published leading coefficient is written `10^7.196` = 1.5705e7, which is
what `C` defaults to. Reading it instead as the literal 107.196 gives departure
diameters near 0.4 um for water, three orders below anything measured, whereas
10^7.196 gives 0.36 mm for water at 10 K superheat and 500 kg/m^2/s - correct to
within the scatter of the data it was fitted to. The exponent SIGNS were
likewise reconstructed on physical grounds (departure shrinks with flow rate and
with density ratio, grows with superheat). Both are worth confirming against the
original before this model is used for a published number.

Sample value for the LH2 pipe (0.4 MPa, `G = 335 kg/m^2/s`, `dT_sup = 1.75 K`):
`L_c = 1.31 nm` and `d_b ~ 47 um`, against 107 um from the departure model
currently in use - and below the ~74 um turbulent-breakup limit at the first
cell, which the larger value exceeds.
"""
struct Du{F<:AbstractFloat} <: AbstractDepartureDiameter
    G::F
    C::F
    d_min::F
    d_max::F
end
Du(; G, C=1.5705e7, d_min=1.0e-8, d_max=1.4e-2) =
    Du(float(G), float(C), float(d_min), float(d_max))
Adapt.@adapt_structure Du

@inline function bubble_departure_diameter(model::Du, s::BoilingState{F}) where F
    s.dT_sup <= zero(F) && return F(model.d_min)
    (s.sigma <= zero(F) || s.rho_l <= zero(F) || s.rho_v <= zero(F) ||
     s.mu_l <= zero(F) || s.k_l <= zero(F) || s.h_fg <= zero(F) ||
     model.G <= zero(F)) && return F(model.d_min)

    nu_l = s.mu_l/s.rho_l
    L_c  = s.rho_l*nu_l*nu_l/s.sigma
    rho_star = s.rho_v/s.rho_l
    Ja = s.cp_l*s.dT_sup/s.h_fg
    Pr = s.mu_l*s.cp_l/s.k_l
    (L_c <= zero(F) || Ja <= zero(F) || Pr <= zero(F)) && return F(model.d_min)

    A = L_c*F(model.C)*rho_star^F(-0.319)*Ja^F(0.123)*Pr^F(-1.939)*
        (F(model.G)/s.mu_l)^F(-0.751)
    A <= zero(F) && return F(model.d_min)
    d = A^(one(F)/F(1.751))
    return clamp(d, F(model.d_min), F(model.d_max))
end


# =============================================================================
#  Bubble departure frequency
# =============================================================================

"""
    bubble_departure_frequency(model, state, D_d) -> 1/s

Rate at which bubbles leave a single active site.
"""
function bubble_departure_frequency end

"""
    Cole(; C_d=1.0, f_max=1.0e4)

Cole (1960) rise-velocity departure frequency,

    f = sqrt( 4*g*(rho_l - rho_v) / (3*C_d*rho_l*D_d) )

obtained by dividing the terminal rise velocity of the bubble by its own
diameter. `C_d` is the drag coefficient (unity in the original).

`f_max` caps the frequency. It matters because `f` diverges as `D_d -> 0`,
which happens at high subcooling with `TolubinskyKostanchuk`, and an unbounded
frequency would make the quenching flux unbounded too.
"""
struct Cole{F<:AbstractFloat} <: AbstractDepartureFrequency
    C_d::F
    f_max::F
end
Cole(; C_d=1.0, f_max=1.0e4) = Cole(float(C_d), float(f_max))
Adapt.@adapt_structure Cole

@inline function bubble_departure_frequency(model::Cole, s::BoilingState{F}, D_d) where F
    drho = s.rho_l - s.rho_v
    (drho <= zero(F) || D_d <= zero(F)) && return zero(F)
    f = sqrt(4*s.g*drho/(3*F(model.C_d)*s.rho_l*D_d))
    return min(f, F(model.f_max))
end


"""
    BaldGrowth(; C_w=0.0, f_max=1.0e4)

Departure frequency from the bubble GROWTH TIME, using the growth law Bald
(1976) validated for liquid hydrogen and liquid helium,

    D_d = 4*sqrt(3/pi) * B * sqrt(a_l*tau_g)

    B   = rho_l*cp_l*dT_sup / ( rho_v*(h_fg + (cp_l - cp_v)*dT_sup) )
    a_l = k_l/(rho_l*cp_l)                          liquid thermal diffusivity

Inverted for the growth time and combined with the waiting time through the
usual `tau_w = C_w/f` closure (Kuang et al. 2021, Eq. 16), `f = 1/(tau_w +
tau_g)` collapses to a closed form with no iteration:

    f = (1 - C_w) * 16*(3/pi) * B^2 * a_l / D_d^2

### Why prefer it to `Cole` for hydrogen

[`Cole`](@ref) divides a bubble's terminal RISE velocity by its diameter, so it
is a pool-boiling buoyancy balance and carries no thermal information at all.
Bald's constant was measured on liquid hydrogen and liquid helium specifically,
and the frequency it gives is set by how fast the bubble can grow on the
superheat available - which is the mechanism that actually limits departure in
saturated flow boiling. Kuang et al. found `Cole` "tends to underestimate the
frequency", and their contrast model built on it gave a mean absolute error of
52.5% against 8.94% for the growth-time route.

On the LH2 pipe at `dT_sup = 1.75 K` and `D_d = 47.7 um`: `B = 1.045`,
`a_l = 8.62e-8 m^2/s`, `tau_g = 1.58 ms`, `f = 633 Hz`, against 503 Hz from
`Cole` - about 26% higher, same order.

### On `C_w`

`C_w` is the fraction of the bubble cycle spent WAITING rather than growing.
The physical route to it is Han & Griffith, which needs the critical cavity
radius `r_c` - a surface property that is rarely known. The default of zero
takes the saturated-flow-boiling limit, which is what Kuang et al. measure:
they report `C_w < 0.1` throughout, and correspondingly find quenching
contributes under 3% of the wall flux. Set it non-zero only with a value you
can defend; it scales `f` linearly and therefore scales `q_evap` linearly.

`cp_v` enters the sensible-heat correction to `h_fg`. `BoilingState` defaults it
to zero, which overstates that correction by a few percent; the solver
populates it from the vapour phase, so this only matters when constructing a
state by hand.
"""
struct BaldGrowth{F<:AbstractFloat} <: AbstractDepartureFrequency
    C_w::F
    f_max::F
end
BaldGrowth(; C_w=0.0, f_max=1.0e4) = BaldGrowth(float(C_w), float(f_max))
Adapt.@adapt_structure BaldGrowth

@inline function bubble_departure_frequency(model::BaldGrowth, s::BoilingState{F}, D_d) where F
    (D_d <= zero(F) || s.dT_sup <= zero(F)) && return zero(F)
    (s.rho_v <= zero(F) || s.rho_l <= zero(F) || s.cp_l <= zero(F) ||
     s.k_l <= zero(F)) && return zero(F)
    C_w = F(model.C_w)
    C_w >= one(F) && return zero(F)

    # Effective latent heat: the bubble must also supply the sensible heat that
    # the displaced liquid carried. Denominator is positive whenever h_fg is.
    h_eff = s.h_fg + (s.cp_l - s.cp_v)*s.dT_sup
    h_eff <= zero(F) && return zero(F)

    B = s.rho_l*s.cp_l*s.dT_sup/(s.rho_v*h_eff)
    a_l = s.k_l/(s.rho_l*s.cp_l)

    # (4*sqrt(3/pi))^2 = 16*3/pi
    f = (one(F) - C_w)*F(16)*F(3)/F(pi)*B*B*a_l/(D_d*D_d)
    return min(f, F(model.f_max))
end


# =============================================================================
#  Bubble influence area
# =============================================================================

"""
    bubble_influence_fraction(model, state, N_a, D_d) -> [-]

Fraction of the wall area affected by bubble growth and departure, i.e. the
area over which quenching replaces single-phase convection. Always in [0, 1].
"""
function bubble_influence_fraction end

"""
    _saturate_area(x, saturation) -> A_b in [0, 1)

How the raw coverage `x = K N_a pi D_d^2/4` is limited to a physical area
fraction. Dispatched on `Val` so the branch resolves at compile time and the
kernel stays branch-free.

### `Val(:clamp)` - `A_b = min(1, x)`

The classical Kurul & Podowski form. It assumes bubble influence zones TILE the
wall without overlapping, so coverage grows linearly until the wall is full.

It has a serious consequence that is not obvious from the formula. Once the clamp
binds, `dA_b/dT_w` is exactly zero and

    q_conv = h_c (T_w - T_l) (1 - A_b) = 0

*identically*, and stays zero for every higher flux. The partition loses the one
term that responds smoothly and linearly to the local liquid temperature, and the
whole wall flux is left to `q_evap ~ dT_sup^n`. Measured on the LH2 pipe with the
calibrated `n = 21.17`, the clamp binds between 36 and 38 kW/m^2 and the local
slope `d(ln q)/d(ln dT_sup)` jumps discontinuously from 7.9 to 18.1 there.

### `Val(:exponential)` - `A_b = 1 - exp(-x)`

Poisson void probability: if influence zones of expected total coverage `x` are
placed at RANDOM on the wall, the fraction left uncovered is `exp(-x)`. Bubbles
nucleate at cavities that are not arranged to tile neatly, so overlap is the
expected behaviour and the linear form is the special case, not this one.

`A_b` then approaches unity asymptotically and never reaches a clamp, so
`q_conv` decays smoothly instead of switching off. On the same LH2 case it
retains 6% of the flux at CHF, the slope discontinuity disappears (the sequence
runs 7.0, 9.0, 9.9, 10.7, 11.4, 13.0, 14.3, 16.1, 16.6) and the peak stiffness
drops from 19.3 to 16.7. The predicted superheat at CHF moves by 0.2%, so an
existing calibration transfers essentially unchanged.

Prefer `:exponential` for anything approaching CHF. `:clamp` remains the default
only so that existing results do not move silently.
"""
@inline _saturate_area(x::F, ::Val{:clamp}) where F = clamp(x, zero(F), one(F))
@inline _saturate_area(x::F, ::Val{:exponential}) where F = -expm1(-max(x, zero(F)))

function _check_saturation(saturation)
    saturation in (:clamp, :exponential) || throw(ArgumentError(
        "`saturation` must be :clamp or :exponential, got :$saturation"))
    return Val(saturation)
end

"""
    DelValleKenning(; K_ref=4.8, dT_ref=80.0, saturation=:clamp)

Del Valle & Kenning (1985) influence area,

    A_b = saturate(K * N_a * pi * D_d^2 / 4),   K = K_ref * exp(-dT_sub/dT_ref)

The influence area is larger than the bubble footprint (`K > 1`) because the
disturbance extends beyond the bubble itself, and it shrinks with subcooling.

Limiting `A_b` to unity is essential, not cosmetic: with `N_a ~ dT_sup^n` the
raw expression exceeds one at quite moderate superheat, at which point the
convective term would go negative and the partition would stop making sense.

HOW it is limited matters as much as that it is - see [`_saturate_area`](@ref).
`:clamp` reproduces the classical model and switches `q_conv` off abruptly;
`:exponential` accounts for overlap between influence zones and does not.
"""
struct DelValleKenning{F<:AbstractFloat,S} <: AbstractInfluenceArea
    K_ref::F
    dT_ref::F
    saturation::Val{S}
end
DelValleKenning(; K_ref=4.8, dT_ref=80.0, saturation=:clamp) =
    DelValleKenning(float(K_ref), float(dT_ref), _check_saturation(saturation))
Adapt.@adapt_structure DelValleKenning

@inline function bubble_influence_fraction(
    model::DelValleKenning, s::BoilingState{F}, N_a, D_d) where F
    K = F(model.K_ref)*exp(-max(s.dT_sub, zero(F))/F(model.dT_ref))
    return _saturate_area(K*N_a*F(pi)*D_d^2/4, model.saturation)
end

"""
    ConstantInfluenceArea(; K=2.0, saturation=:clamp)

Fixed influence factor, `A_b = saturate(K * N_a * pi * D_d^2/4)`.

`K = 2` is the original RPI value (Kurul & Podowski). Useful as a control when
assessing how much of a result comes from the subcooling dependence in
[`DelValleKenning`](@ref). See [`_saturate_area`](@ref) for `saturation`.
"""
struct ConstantInfluenceArea{F<:AbstractFloat,S} <: AbstractInfluenceArea
    K::F
    saturation::Val{S}
end
ConstantInfluenceArea(; K=2.0, saturation=:clamp) =
    ConstantInfluenceArea(float(K), _check_saturation(saturation))
Adapt.@adapt_structure ConstantInfluenceArea

@inline bubble_influence_fraction(
    model::ConstantInfluenceArea, s::BoilingState{F}, N_a, D_d) where F =
    _saturate_area(F(model.K)*N_a*F(pi)*D_d^2/4, model.saturation)


# =============================================================================
#  The RPI model itself
# =============================================================================

"""
    RPI(; site_density, departure_diameter, departure_frequency, influence_area,
          patches, Pr_t=0.85, alpha_min=0.1, n_iterations=40)

Rensselaer Polytechnic Institute wall heat flux partitioning model for
subcooled and saturated nucleate boiling.

Every empirical ingredient is a separate, swappable sub-model; the defaults are
the combination used in most commercial implementations.

### Keywords
- `site_density`        -- [`LemmertChawla`](@ref) (default) or [`HibikiIshii`](@ref).
- `departure_diameter`  -- [`TolubinskyKostanchuk`](@ref) (default) or
                           [`KocamustafaogullariIshii`](@ref).
- `departure_frequency` -- [`Cole`](@ref) (default).
- `influence_area`      -- [`DelValleKenning`](@ref) (default) or
                           [`ConstantInfluenceArea`](@ref).
- `patches`  -- names of the heated wall patches the model applies to, as a
                tuple of `Symbol`s, e.g. `(:pipeWall,)`. Each must carry a
                `FixedHeatFlux` boundary condition on `T`.
- `Pr_t`     -- turbulent Prandtl number for the thermal wall function.
- `alpha_min`-- liquid volume fraction below which wall boiling is switched off
                (see below).
- `n_iterations` -- bisection steps for the wall temperature solve.

### How it couples to the solver

The model does **not** replace the temperature boundary condition. The imposed
`FixedHeatFlux` still delivers the full `q_w` into the near-wall cell; what RPI
adds is the vapour generation rate

    mdot_wall = q_e * A_face / (h_fg * V_cell)      [kg/m^3/s]

which enters the same three places as the bulk phase change rate - the volume
fraction equation, the pressure equation's volume creation, and the energy
equation's latent heat sink. The latent heat sink is what removes `q_e` from the
liquid again, so the net heating of the liquid is `q_c + q_q` without the
boundary condition having to know anything about boiling.

### Dryout

RPI assumes a liquid-wetted wall. Above a wall vapour fraction the model has no
validity, and continuing to apply it would keep generating vapour from liquid
that is not there. Vapour generation is therefore ramped linearly to zero as the
near-wall liquid fraction falls from `2*alpha_min` to `alpha_min`. This is a
numerical safeguard, not a dryout model: it prevents an unphysical source, but
it does not predict the post-DNB regime, and a case that spends real time in the
ramp is outside what this implementation can claim.

### Example
```julia
wall_boiling = RPI(
    site_density = LemmertChawla(),          # N_a = (210 dT_sup)^1.805
    departure_diameter = TolubinskyKostanchuk(d_ref=0.6e-3, dT_ref=45.0),
    departure_frequency = Cole(),
    influence_area = DelValleKenning(),
    patches = (:pipeWall,),
)
```
"""
struct RPI{S,D,Fr,A,P,BL,H,F<:AbstractFloat} <: AbstractWallBoilingModel
    site_density::S
    departure_diameter::D
    departure_frequency::Fr
    influence_area::A
    patches::P
    Pr_t::F
    alpha_min::F
    dryout_start::F
    dryout_end::F
    bubbly_layer::BL
    dryout_smoothing::Int
    dryout_smoothing_weight::F
    dryout_filter::Symbol
    dryout_relaxation::F
    dryout_shape::Symbol
    dryout_snap::F
    hysteresis::H
    wall_capacity::F
    n_iterations::Int
    start_iteration::Int
    friction_velocity::Symbol
    partition::Symbol
end
Adapt.@adapt_structure RPI

function RPI(;
    site_density = LemmertChawla(),
    departure_diameter = TolubinskyKostanchuk(),
    departure_frequency = Cole(),
    influence_area = DelValleKenning(),
    patches,
    Pr_t = 0.85,
    alpha_min = 0.1,
    dryout_start = nothing,
    dryout_end = nothing,
    bubbly_layer = nothing,
    dryout_smoothing = 0,
    dryout_smoothing_weight = 0.5,
    dryout_filter = :median,
    dryout_relaxation = 1.0,
    dryout_shape = :smoothstep,
    dryout_snap = 1.0,
    hysteresis = nothing,
    wall_capacity = 0.0,
    n_iterations = 40,
    start_iteration = 0,
    friction_velocity = :k,
    partition = :kurul_podowski)

    patches_tuple = patches isa Symbol ? (patches,) : Tuple(patches)
    isempty(patches_tuple) && throw(ArgumentError(
        "`RPI` needs at least one heated wall patch, e.g. `patches = (:pipeWall,)`"))
    all(p -> p isa Symbol, patches_tuple) || throw(ArgumentError(
        "`patches` must be boundary names as Symbols, got $(patches)"))

    wall_capacity >= 0 || throw(ArgumentError(
        "`wall_capacity` must be non-negative, got $wall_capacity. Zero means no wall inertia."))

    start_iteration >= 0 || throw(ArgumentError(
        "`start_iteration` must be non-negative, got $start_iteration."))

    # `:k`      u_tau = Cmu^0.25*sqrt(k)   - assumes LOCAL EQUILIBRIUM in the
    #                                        near-wall cell. Existing default.
    # `:loglaw` Newton solve of the log law from the VELOCITY - the same equation
    #                                        the momentum wall treatment uses, and
    #                                        valid whether or not k has settled.
    #
    # `h_c = rho*cp*u_tau/T+` is linear in u_tau, so this choice scales the
    # convective share of the RPI partition directly. A boiling wall disturbs the
    # near-wall balance that `:k` assumes, which is why `:loglaw` may do better
    # there - see `_u_tau_loglaw!`.
    # DRYOUT RAMP. `nothing` reproduces the legacy behaviour exactly: a LINEAR
    # ramp derived from `alpha_min`, over void `1-2*alpha_min` to `1-alpha_min`
    # (0.8 to 0.9 at the default). Supplying BOTH gives a SMOOTHSTEP over an
    # explicit void interval - see `wall_boiling_liquid_factor`.
    (dryout_start === nothing) == (dryout_end === nothing) || throw(ArgumentError(
        "`dryout_start` and `dryout_end` must be given together, or neither"))
    ds = dryout_start === nothing ? -one(float(alpha_min)) : float(dryout_start)
    de = dryout_end   === nothing ? -one(float(alpha_min)) : float(dryout_end)
    dryout_shape in (:smoothstep, :step) || throw(ArgumentError(
        "`dryout_shape` must be :smoothstep or :step, got :$dryout_shape"))

    # `dryout_snap` is the K_dry above which the ramp completes at once. 1 is the
    # plain smoothstep - see `_dryout_shape`.
    0 < dryout_snap <= 1 || throw(ArgumentError(
        "`dryout_snap` must be in (0, 1], got $dryout_snap"))

    if dryout_start !== nothing
        if dryout_shape === :step
            # `:step` switches at `dryout_start` and never reads `dryout_end`, so
            # the two may coincide - and `dryout_end = dryout_start` is the
            # natural way to write "no ramp".
            0 <= ds <= 1 || throw(ArgumentError(
                "need 0 <= dryout_start <= 1, got $ds"))
            de >= ds || throw(ArgumentError(
                "`dryout_end` must not be below `dryout_start`, got $de and $ds"))
        else
            0 <= ds < de <= 1 || throw(ArgumentError(
                "need 0 <= dryout_start < dryout_end <= 1, got $ds and $de"))
        end
    end

    # WALL-TANGENTIAL SMOOTHING of the bubbly-layer void before it reaches the
    # dryout ramp. Every other coupling in this model is wall-NORMAL: one
    # independent bisection per face, one normal extrapolation per face, and a
    # pointwise `K_dry`. Nothing ties a face to the ones beside it, so a steep
    # closure lets neighbours settle on different branches - which has shown up
    # as azimuthal void scatter, isolated fully-dry faces, and a jagged `K_dry`
    # front. See `build_wall_face_graph`. Zero passes disables it entirely.
    dryout_smoothing >= 0 || throw(ArgumentError(
        "`dryout_smoothing` must be non-negative, got $dryout_smoothing"))
    0 < dryout_smoothing_weight <= 1 || throw(ArgumentError(
        "`dryout_smoothing_weight` must be in (0, 1], got $dryout_smoothing_weight"))

    # WHICH FILTER. `:median` is the default and the one to use.
    #
    # `:laplacian` averages a face toward its neighbours, which is a PEAK KILLER -
    # and a localised dry patch IS a peak. Measured on the LH2 pipe: at the step
    # where the void ran away, two Laplacian passes suppressed the `alpha_delta`
    # peak by 0.35 (raw 1.00 -> 0.65) while leaving the mean unchanged to 0.03%,
    # halving `K_dry` (0.594 -> 0.277) exactly when dryout needed to fire. In
    # quiet states the same filter changed the peak by 0.002, which is why it
    # looked harmless in every check that did not span a runaway.
    #
    # `:median` removes ISOLATED face-to-face outliers - the odd-even mode that
    # motivated smoothing in the first place - while preserving a coherent front,
    # because it returns an actual neighbour value rather than an average. A
    # single dry face among wet neighbours is removed; a dry FRONT is not.
    #
    # `:laplacian` is kept only so the earlier behaviour can be reproduced.
    dryout_filter in (:median, :laplacian) || throw(ArgumentError(
        "`dryout_filter` must be :median or :laplacian, got :$dryout_filter"))

    # TEMPORAL RELAXATION of alpha_delta - the gain limiter on the dryout loop.
    #
    #     alpha_delta <- (1-r)*alpha_delta_prev + r*alpha_delta_new
    #
    # `K_dry` closes a NEGATIVE feedback loop (more void -> more dryout -> less
    # evaporation -> less void). Self-correcting at low gain, but it OSCILLATES
    # once the loop gain exceeds 1 with a step of delay - and the gain is large.
    # `K_dry = smoothstep((a - ds)/(de - ds))` has slope `6b(1-b)/(de-ds)`,
    # peaking at `1.5/(de-ds)`: that is 3.0 for the usual 0.5..1.0 ramp, reached
    # at a = 0.75. Against `q_evap + q_quench ~ 5.6e4 W/m^2` that is 1.7e5 W/m^2
    # per unit void, so a 0.1 wobble swings a quarter of the applied flux.
    #
    # Measured on the LH2 pipe at 7e4: the wall balance held to 0.03% of applied
    # for 1500 steps with alpha_delta at 0.58 (gain 1.7), then lost it entirely -
    # q_total spanning 3% to 211% of applied - within the 500 steps it took
    # alpha_delta to sweep through 0.75, where the gain peaks.
    #
    # `r` multiplies the high-frequency loop gain directly, so `r < (de-ds)/1.5`
    # brings it under unity. It does NOT change the converged answer: at steady
    # state new == prev and the blend is the identity.
    #
    # NOT the same as lagging the field, which was tried and diverged. A lag
    # substitutes an older value - pure phase shift, no gain reduction - and phase
    # is what destabilises this loop. Relaxation ATTENUATES instead.
    #
    # `r = 1` is no relaxation and reproduces the previous behaviour exactly.
    0 < dryout_relaxation <= 1 || throw(ArgumentError(
        "`dryout_relaxation` must be in (0, 1], got $dryout_relaxation"))

    # `:step` switches at `dryout_start` and ignores `dryout_end` - see
    # `_dryout_shape` for why the ramp is inert under a flux-controlled wall.

    # WHICH PARTITION. See `wall_heat_partition`.
    partition in (:kurul_podowski, :mmp) || throw(ArgumentError(
        "`partition` must be :kurul_podowski or :mmp, got :$partition"))

    friction_velocity in (:k, :loglaw) || throw(ArgumentError(
        "`friction_velocity` must be :k or :loglaw, got :$friction_velocity"))


    return RPI(site_density, departure_diameter, departure_frequency, influence_area,
               patches_tuple, float(Pr_t), float(alpha_min),
               ds, de, bubbly_layer, dryout_smoothing,
               float(dryout_smoothing_weight), dryout_filter,
               float(dryout_relaxation), dryout_shape, float(dryout_snap),
               hysteresis,
               float(wall_capacity), n_iterations, start_iteration, friction_velocity,
               partition)
end

wall_boiling_patches(model::RPI) = model.patches

"""
    wall_boiling_active(model, iteration) -> Bool

Whether wall boiling should contribute at this iteration.

`start_iteration` holds the model off until the base flow is established. This is
NOT cosmetic: `h_c` is built from the turbulence model's friction velocity, and
`T_w = T_l + q_w/h_c` follows from it, so engaging RPI while `k`, `omega` and the
velocity profile are still at their uniform initial values feeds the site density
(~dT_sup^1.805) a wall temperature derived from a flow that does not yet exist.

It is also a diagnostic. A case that is stable with a delay and unstable without
one has a STARTUP problem; a case that fails either way has a genuine
instability, and the two need different fixes.

Zero (the default) engages from the first iteration, preserving existing
behaviour.
"""
wall_boiling_active(model::RPI, iteration) = iteration > model.start_iteration
wall_boiling_active(::Nothing, iteration) = false


# =============================================================================
#  Heat flux partition
# =============================================================================

"""
    wall_heat_partition(rpi, state, h_c) -> (q_c, q_q, q_e, A_b, N_a, D_d, f)

The three RPI flux components [W/m^2] at the wall temperature carried by
`state`, plus the intermediate bubble quantities.

`h_c` is the single-phase convective heat transfer coefficient at the wall
(see [`single_phase_htc`](@ref)).

    q_c = h_c (T_w - T_l) (1 - A_b)
    q_q = 2 f sqrt(t_w k_l rho_l cp_l / pi) (T_w - T_l) A_b,   t_w = 0.8/f
    q_e = N_a f (pi/6) D_d^3 rho_v h_fg

Below saturation `N_a`, and with it `A_b`, is zero, so the expression collapses
to pure single-phase convection - the correct non-boiling limit, and the reason
the same routine can be used on both sides of onset.
"""
@inline function wall_heat_partition(rpi::RPI, s::BoilingState{F}, h_c) where F
    dT_wl = s.T_w - s.T_l

    N_a = nucleation_site_density(rpi.site_density, s)
    D_d = bubble_departure_diameter(rpi.departure_diameter, s)
    f = bubble_departure_frequency(rpi.departure_frequency, s, D_d)
    A_b = bubble_influence_fraction(rpi.influence_area, s, N_a, D_d)

    # Quenching kernel, before any area or dryout weighting. Mikic & Rohsenow
    # transient conduction over the waiting time t_w = 0.8/f; writing it as
    # sqrt(t_w * k rho cp) keeps the group in terms of the thermal effusivity and
    # avoids dividing by the diffusivity.
    q_q_raw = if f > zero(F)
        t_w = F(0.8)/f
        2*f*sqrt(t_w*s.k_l*s.rho_l*s.cp_l/F(pi))*dT_wl
    else
        zero(F)
    end
    q_e_raw = N_a*f*(F(pi)/6)*D_d^3*s.rho_v*s.h_fg

    q_c, q_q, q_e = _partition_weights(
        Val(rpi.partition === :mmp), rpi, s, h_c, dT_wl, A_b, q_q_raw, q_e_raw)

    return (q_c=q_c, q_q=q_q, q_e=q_e, A_b=A_b, N_a=N_a, D_d=D_d, f=f)
end

# KURUL & PODOWSKI (default). Convection over the un-influenced wall, quenching
# over the bubble-influenced part, evaporation unweighted:
#
#     q_w = h_c dT (1 - A_b) + q_quench A_b + q_evap
#
@inline _partition_weights(::Val{false}, rpi, s::BoilingState{F}, h_c, dT_wl,
                           A_b, q_q_raw, q_e_raw) where F =
    (h_c*dT_wl*(one(F) - A_b), q_q_raw*A_b, q_e_raw)

# STAR-CCM+ MIXTURE MULTIPHASE (`partition = :mmp`), User Guide Eqn (2944):
#
#     q_w = q_conv + (q_evap + q_quench)(1 - K_dry)
#
# TWO differences from Kurul-Podowski, both deliberate:
#
#   1. `q_conv` is NOT area-weighted by `A_b`. STAR-CCM+ treats it as the MIXTURE
#      convection of whatever is in contact with the wall - "there [are]
#      convection contributions from vapor and liquid, always the mixture in
#      contact with the wall" - handled by the energy model rather than split off
#      a liquid-wetted fraction. The caller supplies a MIXTURE `h_c` to match.
#
#   2. `K_dry`, the wall dryout area fraction, multiplies `q_evap` AND
#      `q_quench` but NOT `q_conv`. Here `K_dry = 1 - wall_boiling_liquid_factor`,
#      reusing the existing ramp so the two formulations share one threshold.
#
# WHY IT MATTERS HERE. Under Kurul-Podowski `q_c` keeps using LIQUID properties
# however dry the wall gets, and the vapour source is cut separately AFTER the
# wall temperature solve - so at high void the wall has no valid convective path
# and the flux is dumped through the boundary condition as sensible heat. Under
# `:mmp` the convective term degrades continuously into vapour convection and,
# because `(1 - K_dry)` sits INSIDE the inversion, dryout RAISES the wall
# temperature, which is the physical effect STAR-CCM+ describes: "vapor heat
# transfer removes some fraction of the wall heat flux and causes an increase in
# wall temperature".
@inline function _partition_weights(::Val{true}, rpi, s::BoilingState{F}, h_c,
                                    dT_wl, A_b, q_q_raw, q_e_raw) where F
    one_minus_Kdry = wall_boiling_liquid_factor(rpi, s.alpha_l)
    return (h_c*dT_wl, q_q_raw*A_b*one_minus_Kdry, q_e_raw*one_minus_Kdry)
end

"""Sum of a partition, whichever form produced it."""
@inline _partition_total(p::NamedTuple{(:q_c,:q_q,:q_e,:A_b,:N_a,:D_d,:f)}) =
    p.q_c + p.q_q + p.q_e

"""
    solve_wall_temperature(rpi, state, q_w, h_c) -> (T_w, partition)

Invert the RPI partition: find the wall temperature at which the three
components sum to the imposed wall heat flux `q_w`.

The experiment being reproduced (Tatsumoto et al., 2014) prescribes the heat
*generation rate* in the tube, so the flux is the known quantity and the wall
temperature is what the model must predict - the wall superheat then follows,
and it is what every sub-model above keys off.

### Method

Bisection on a bracket that is guaranteed by construction:

- **lower bound** `T_sat`: at saturation there are no active sites, so the total
  is just `h_c (T_sat - T_l)`, which is at most `q_w` whenever boiling occurs;
- **upper bound** `T_l + q_w/h_c`: the pure-convection wall temperature. Adding
  the two non-negative boiling terms can only reduce the superheat needed, so
  the true root never exceeds it.

Total flux is monotonically increasing in `T_w` across that interval, so
bisection converges unconditionally. It is used in preference to Newton because
`N_a ~ dT_sup^1.805` makes the residual extremely stiff near onset, where a
Newton step readily overshoots into negative superheat; and because a fixed
iteration count is branch-free, which matters for a GPU kernel.

If the pure-convection wall temperature is already at or below `T_sat` there is
no boiling and it is returned directly.
"""
@inline function solve_wall_temperature(rpi::RPI, s::BoilingState{F}, q_w, h_c) where F
    # Non-boiling: single-phase convection carries the whole flux.
    T_w_conv = s.T_l + q_w/max(h_c, eps(F))
    if T_w_conv <= s.T_sat || q_w <= zero(F)
        s_conv = _at_wall_temperature(s, T_w_conv)
        return (T_w_conv, wall_heat_partition(rpi, s_conv, h_c))
    end

    lo = s.T_sat
    hi = T_w_conv

    # EXPAND the upper bound until the partition there actually reaches `q_w`.
    #
    # `T_w_conv = T_l + q_w/h_c` is NOT a guaranteed bound, contrary to the
    # argument that adding non-negative boiling terms can only lower the
    # superheat. It ignores `A_b`: at that temperature
    #
    #     q_conv = h_c*dT*(1 - A_b) = q_w*(1 - A_b)  <  q_w
    #
    # so unless quenching and evaporation cover the `A_b` deficit, the root lies
    # ABOVE the bracket and bisection saturates at the top, returning a partition
    # that does not sum to `q_w`. The failure is silent - the wall temperature
    # simply comes out too low.
    #
    # It went unnoticed while `single_phase_htc` returned a `T+` that was ~3x too
    # large: the resulting small `h_c` put the bracket top far above `T_sat`,
    # where `q_evap ~ dT_sup^n` is enormous and always covered the shortfall.
    # Correcting `T+` moved the top close to `T_l` and exposed it.
    #
    # Doubling the superheat 8 times covers 256x, far beyond any physical root.
    # Written with a ternary rather than a `break` so the iteration count is
    # fixed and the kernel stays branch-free.
    for _ in 1:8
        p_hi = wall_heat_partition(rpi, _at_wall_temperature(s, hi), h_c)
        short = (p_hi.q_c + p_hi.q_q + p_hi.q_e) < q_w
        hi = short ? s.T_l + 2*(hi - s.T_l) : hi
    end

    for _ in 1:rpi.n_iterations
        mid = (lo + hi)/2
        s_mid = _at_wall_temperature(s, mid)
        p = wall_heat_partition(rpi, s_mid, h_c)
        total = p.q_c + p.q_q + p.q_e
        if total < q_w
            lo = mid
        else
            hi = mid
        end
    end

    T_w = (lo + hi)/2
    return (T_w, wall_heat_partition(rpi, _at_wall_temperature(s, T_w), h_c))
end


# =============================================================================
#  Single-phase wall heat transfer coefficient
# =============================================================================

"""
    solve_wall_temperature_transient(rpi, state, q_gen, h_c, T_w_prev, dt)

Wall temperature with **thermal inertia**: instead of inverting the partition
algebraically, integrate the lumped wall energy balance

    C dT_w/dt = q_gen - [q_conv(T_w) + q_quench(T_w) + q_evap(T_w)]

where `C = rho_w * cp_w * thickness` is `rpi.wall_capacity` [J/m^2/K].

### Why this matters

[`solve_wall_temperature`](@ref) makes `T_w` respond **instantaneously** to any
change in the near-wall liquid temperature or heat transfer coefficient. With
`N_a ~ dT_sup^1.805` that response is violently sensitive, and there is nothing
anywhere in the boundary treatment with a time constant.

A real heater has one. For the Tatsumoto tubes - SS316, 0.5 mm wall - the
capacity is `rho*cp*delta`. Note `cp` for stainless at 30 K is around
20 J/kg/K, an order of magnitude below its room-temperature value (Debye T^3),
so `C ~ 8000 * 20 * 0.5e-3 ~ 80 J/m^2/K`. Against 3e4 W/m^2 that gives a time
constant of ~13 ms to move `T_w` by 5 K - some 6500 steps at `dt = 2e-6 s`,
rather than one.

It also restores the physical negative feedback the algebraic form lacks: if the
fluid cannot absorb the heat, the wall temperature rises and the partition
shifts, instead of the flux being imposed regardless of what the fluid does.

### Method

Implicit (backward) Euler, so the stiff `T_w` dependence is evaluated at the new
time level:

    f(T_w) = C(T_w - T_w_prev)/dt + Q(T_w) - q_gen = 0

`f` is monotonically increasing - both the storage and the flux terms grow with
`T_w` - so bisection converges unconditionally, as in the steady inversion.

`C = 0` recovers [`solve_wall_temperature`](@ref) exactly, which is the default
and keeps the previous behaviour unchanged.
"""
@inline function solve_wall_temperature_transient(
    rpi::RPI, s::BoilingState{F}, q_gen, h_c, T_w_prev, dt) where F

    C = F(rpi.wall_capacity)
    # No inertia, or no usable previous state (first step): fall back to the
    # steady inversion, which is also what seeds `T_w_prev`.
    (C <= zero(F) || !(T_w_prev > zero(F))) &&
        return _steady_fallback(rpi, s, q_gen, h_c)

    Cdt = C/F(dt)

    # f increases with T_w. Lower bound: the coldest state in play, where Q -> 0
    # and the storage term is at its most negative. Upper bound: the temperature
    # the wall would reach on storage alone with Q = 0, which cannot be exceeded.
    #
    # `f` is monotone in `T_w`: every partition term rises with wall temperature
    # and the storage term `Cdt*(T_w - T_w_prev)` rises with it, so bisection is
    # unconditional.
    lo = min(s.T_l, s.T_sat, T_w_prev)
    hi = max(T_w_prev + q_gen/Cdt, lo) + max(q_gen/max(h_c, eps(F)), zero(F))

    for _ in 1:rpi.n_iterations
        mid = (lo + hi)/2
        p = wall_heat_partition(rpi, _at_wall_temperature(s, mid), h_c)
        f = Cdt*(mid - T_w_prev) + _partition_total(p) - q_gen
        if f < zero(F)
            lo = mid
        else
            hi = mid
        end
    end

    T_w = (lo + hi)/2
    return (T_w, wall_heat_partition(rpi, _at_wall_temperature(s, T_w), h_c))
end

# First step, before `T_w_prev` exists. Seeded from the steady inversion, which
# is well posed and starts the wall on the low branch - where a heated tube
# physically starts.
@inline function _steady_fallback(rpi, s, q_gen, h_c)
    T_w, _ = solve_wall_temperature(rpi, s, q_gen, h_c)
    return (T_w, wall_heat_partition(rpi, _at_wall_temperature(s, T_w), h_c))
end

"""
    single_phase_htc(y_plus, u_tau, rho, cp, mu, k, Pr_t; kappa=0.41, E=9.8)

Single-phase convective heat transfer coefficient at a wall, from the standard
thermal law of the wall:

    h_c = rho * cp * u_tau / T_plus

with the two-layer temperature profile

    T+ = Pr * y+                                   (viscous sublayer)
    T+ = Pr_t * [ln(E y+)/kappa] + P               (log layer)

and the Jayatilleke sublayer resistance

    P = 9.24 [(Pr/Pr_t)^0.75 - 1] [1 + 0.28 exp(-0.007 Pr/Pr_t)]

The blend point is where the two expressions cross, found here by taking the
larger of the two `T+` values - which is equivalent, continuous, and avoids
having to solve for the crossing.

This is the reason the pipe case is meshed for `y+` in the 30-50 range: the log
branch is what is being used, and it is only valid once the first cell centre is
clear of the buffer layer.
"""
@inline function single_phase_htc(y_plus::F, u_tau, rho, cp, mu, k, Pr_t;
                                  kappa=F(0.41), E=F(9.8)) where F<:AbstractFloat
    (k <= zero(F) || u_tau <= zero(F)) && return zero(F)

    Pr = cp*mu/k
    yp = max(y_plus, eps(F))

    T_plus_lam = Pr*yp

    ratio = Pr/Pr_t
    P = F(9.24)*(ratio^F(0.75) - one(F))*(one(F) + F(0.28)*exp(-F(0.007)*ratio))
    T_plus_log = Pr_t*(log(E*yp)/kappa) + P

    # MINIMUM, not maximum. T+ is a thermal RESISTANCE: in the viscous sublayer
    # only conduction acts, giving T+ = Pr*y+, and in the log layer turbulent
    # mixing opens a second transport path that REDUCES the resistance. So the
    # physical T+ is always the smaller of the two branches:
    #
    #   below the crossing   T_plus_lam < T_plus_log   -> laminar is physical
    #   above the crossing   T_plus_lam > T_plus_log   -> log is physical
    #
    # `min` therefore selects correctly in both regimes without solving for the
    # crossing point, which is what this trick is for.
    #
    # This was `max`, which picks the WRONG branch in both. At y+ = 40 with
    # Pr = 1.72 it returned T+ = 68.8 (laminar) instead of 20.6 (log), making
    # `h_c = rho*cp*u_tau/T+` a factor of 3.3 too SMALL - so the RPI convective
    # share was starved and the partition made up the difference through
    # evaporation, inflating the wall superheat. Measured h_conv of
    # 3692-5388 W/m^2/K against a Dittus-Boelter estimate of ~10,200 is that
    # factor almost exactly.
    T_plus = min(T_plus_lam, T_plus_log)
    T_plus <= zero(F) && return zero(F)

    return rho*cp*u_tau/T_plus
end

"""
    wall_boiling_liquid_factor(rpi, alpha_l)

Smooth cut-off applied to the wall vapour generation rate as the near-wall
liquid runs out. Unity above `2*alpha_min`, zero below `alpha_min`, linear
between. See the dryout note in the [`RPI`](@ref) docstring.
"""
# Cubic smoothstep, C1 at both ends. Lived in `2_film_boiling_models.jl` until
# the film branch was removed; `wall_boiling_liquid_factor` is now its only
# caller, so it belongs here. NOTE this is called from inside an `@inline`, so a
# missing definition compiles cleanly and only fails when dryout first evaluates.
@inline _smoothstep(x::F) where F =
    (y = clamp(x, zero(F), one(F)); y*y*(3 - 2*y))

"""
    _dryout_shape(shape, void, ds, de) -> K_dry

`:smoothstep` ramps over `ds..de`; `:step` switches at `ds` and ignores `de`.

### Why `:step` is the better default for a FLUX-CONTROLLED wall

The ramp region does no work and costs stability, and both halves of that are
measured rather than argued.

NO WORK. Under `FixedHeatFlux` the wall temperature is free, and the solve finds
`T_w` such that the partition sums to the applied flux. `K_dry` is a CONSTANT
during that solve, so suppressing `q_e` by `(1 - K_dry)` merely makes `T_w` rise
until `q_e_raw ~ dT_sup^n` regenerates it. With `n = 9.945` the compensation is
the tenth root: measured at a fixed `h_c` and `q_w = 7e4`,

    K_dry     dT_sup    q_e_raw     q_e = raw*(1-K_dry)
    0.0        1.825    5.06e4      5.06e4
    0.9        2.319    5.49e5      5.49e4     <- HIGHER than no dryout
    0.9999     4.528    4.25e8      4.25e4
    1.0       11.544    -           0          <- the only value that does anything

Four orders of magnitude of suppression cost 2.7 K of superheat. Every value
below 1 is indistinguishable from zero dryout.

COSTS STABILITY. `K_dry` closes a negative feedback loop (void up, dryout up,
evaporation down, void down) whose gain is `dK_dry/d(void)` - which is nonzero
ONLY in the ramp. Measured: a 0.5-0.9 ramp (peak gain 3.75) was stable but capped
`K_dry` at 0.48; narrowing to 0.5-0.645 (gain 10.3) clamped the void LOWER, at
0.62, and made it chatter. The ramp is the entire source of that trade.

`:step` has zero gain everywhere except at the switch, reaches `K_dry = 1` the
moment the criterion is met rather than at the top of a ramp the void may never
climb, and leaves the nucleate branch untouched - because partial dryout was
never doing anything there either.

Pair it with [`PlayHysteresis`](@ref) to get a two-threshold relay: switching at
`ds + r` on the way up and `ds - r` on the way down.
"""
@inline function _dryout_shape(::Val{:smoothstep}, void::F, ds, de, snap) where F
    k = _smoothstep((void - ds)/(de - ds))
    # SNAP. Above `snap` the ramp completes immediately instead of asymptoting.
    #
    # This is what makes a ramp usable at all under a flux-controlled wall. The
    # excursion needs `K_dry` EXACTLY 1 - `0.9999` still leaves the wall on the
    # nucleate branch - and a smoothstep only reaches 1 when the void reaches
    # `dryout_end`, which on this mesh it may never do. The snap moves the point
    # of full dryout to a void the solution can actually attain, while keeping
    # the gradual approach below it.
    #
    # `snap = 1` is the plain smoothstep: `k >= 1` only where `k` is already 1.
    return ifelse(k >= snap, one(F), k)
end
@inline _dryout_shape(::Val{:step}, void::F, ds, de, snap) where F =
    ifelse(void >= ds, one(F), zero(F))
@inline _dryout_shape(shape::Symbol, void::F, ds, de, snap) where F =
    _dryout_shape(Val(shape), void, ds, de, snap)

"""
    dryout_snap_void(rpi) -> void at which K_dry snaps to 1

The void fraction where a `:smoothstep` ramp with `dryout_snap` completes. Useful
for checking the snap lands somewhere the solution actually reaches - solve
`smoothstep(b) = snap` for `b`, then `void = ds + b*(de - ds)`.
"""
function dryout_snap_void(rpi::RPI)
    ds, de, snap = rpi.dryout_start, rpi.dryout_end, rpi.dryout_snap
    ds >= 0 || return NaN
    rpi.dryout_shape === :step && return ds
    lo, hi = 0.0, 1.0
    for _ in 1:60
        mid = 0.5*(lo + hi)
        _smoothstep(mid) < snap ? (lo = mid) : (hi = mid)
    end
    return ds + 0.5*(lo + hi)*(de - ds)
end

"""
    PlayHysteresis(; r)

Rate-independent hysteresis on the dryout criterion, via the PLAY (backlash)
operator - the building block of Prandtl-Ishlinskii hysteresis.

A per-face shadow state `xi` follows the void with a dead zone of half-width `r`:

    xi <- clamp(xi_prev, alpha - r, alpha + r)
    K_dry = smoothstep((xi - dryout_start)/(dryout_end - dryout_start))

so `K_dry` runs along the ramp shifted RIGHT by `r` while the void rises, and
LEFT by `r` while it falls. Two branches from one ramp - no second set of
thresholds to calibrate.

### Why an operator rather than a latch

  * RATE-INDEPENDENT: the result depends on the PATH of `alpha`, not on how fast
    it moved, so it does not change with the time step. That is the defining
    property of physical hysteresis.
  * LIPSCHITZ CONTINUOUS, unlike a relay. `K_dry` never steps, so the vapour
    source `mdot` never steps either - no temporal softening is needed.
  * ORDER-PRESERVING: it cannot manufacture oscillation of its own.

### What it fixes

`K_dry` closes a negative feedback loop - void up, dryout up, evaporation down,
void down - and that ONE loop causes two symptoms: it clamps the void, and above
a gain it oscillates. Measured on the LH2 pipe: a 0.5-0.9 ramp (gain 3.75) was
stable but capped `K_dry` at 0.48, while 0.5-0.645 (gain 10.3) reached higher
`K_dry` but clamped the void LOWER, at 0.62, and chattered. The knob that
sharpens the trigger is the same one that puts the target out of reach.

Inside the play band `dK_dry/dalpha` is EXACTLY ZERO, so an oscillation of
amplitude below `2r` produces no change in `K_dry` at all - the loop is opened,
not merely damped - while the ramp stays as steep as it was. `r` therefore does
not trade against sharpness, which is what separates this from widening the ramp.

### Choosing r

    2r > the numerical chatter amplitude   (measured alpha_delta face-to-face
                                            scatter here: 0.008 to 0.077)
    2r < the physical hysteresis width     (void at DNB minus void at rewetting)

`r = 0.05` satisfies both on this case.

### Chilldown

Path dependence is built in. Seed `xi` high (dry) with a hot wall and the
operator tracks DOWN the upper branch as the void falls, rewetting at a lower
void than the one that dried it. Walk the flux up instead and it traverses the
lower branch. Same operator, same parameter, both directions.
"""
struct PlayHysteresis{F<:AbstractFloat}
    r::F
end
function PlayHysteresis(; r)
    r > 0 || throw(ArgumentError("`r` must be positive, got $r"))
    r < 0.5 || throw(ArgumentError(
        "`r` must be well below 0.5, got $r - the play band is 2r wide and would " *
        "span the whole void range"))
    return PlayHysteresis(float(r))
end
Adapt.@adapt_structure PlayHysteresis

"""
    play_update(::Nothing, xi, alpha) -> alpha
    play_update(h::PlayHysteresis, xi, alpha) -> xi_new

One step of the play operator. With no hysteresis model the state is simply the
input, so every caller can apply it unconditionally.
"""
@inline play_update(::Nothing, xi, alpha) = alpha
@inline play_update(h::PlayHysteresis, xi::F, alpha) where F =
    clamp(xi, alpha - F(h.r), alpha + F(h.r))

@inline function wall_boiling_liquid_factor(rpi::RPI, alpha_l::F) where F
    ds = F(rpi.dryout_start)
    if ds >= zero(F)
        # EXPLICIT VOID INTERVAL, smoothstep - the same shape `VoidTransition`
        # uses for `w_film`, so the dryout ramp and the film blend stop being two
        # differently-shaped descriptions of one transition. With the film branch
        # OFF this IS the transition, which is the STAR-CCM+ MMP arrangement:
        # dryout removes the nucleate terms and mixture convection carries the
        # wall from there.
        de = F(rpi.dryout_end)
        void = one(F) - clamp(alpha_l, zero(F), one(F))
        return one(F) - _dryout_shape(rpi.dryout_shape, void, ds, de,
                                      F(rpi.dryout_snap))
    end
    # LEGACY: linear in the LIQUID fraction between `alpha_min` and `2*alpha_min`.
    lo = F(rpi.alpha_min)
    hi = 2*lo
    return clamp((alpha_l - lo)/(hi - lo + eps(F)), zero(F), one(F))
end
