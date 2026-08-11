export AbstractWallBoilingModel, RPI, wall_boiling_active
export AbstractNucleationSiteDensity, LemmertChawla, HibikiIshii
export AbstractDepartureDiameter, TolubinskyKostanchuk, KocamustafaogullariIshii
export AbstractDepartureFrequency, Cole
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
end
Adapt.@adapt_structure BoilingState

"""
    BoilingState(; T_w, T_l, T_sat, rho_l, rho_v, cp_l, k_l, mu_l, sigma, h_fg, g)

Convenience constructor deriving `dT_sup` and `dT_sub` from the temperatures.
"""
function BoilingState(; T_w, T_l, T_sat, rho_l, rho_v, cp_l, k_l, mu_l, sigma, h_fg, g)
    F = promote_type(typeof(float(T_w)), typeof(float(rho_l)))
    return BoilingState{F}(
        F(T_w), F(T_l), F(T_sat), F(T_w - T_sat), F(T_sat - T_l),
        F(rho_l), F(rho_v), F(cp_l), F(k_l), F(mu_l), F(sigma), F(h_fg), F(g))
end

"""Same state at a different wall temperature. Used by the wall-temperature solve."""
@inline _at_wall_temperature(s::BoilingState{F}, T_w) where F = BoilingState{F}(
    T_w, s.T_l, s.T_sat, T_w - s.T_sat, s.dT_sub,
    s.rho_l, s.rho_v, s.cp_l, s.k_l, s.mu_l, s.sigma, s.h_fg, s.g)


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
    DelValleKenning(; K_ref=4.8, dT_ref=80.0)

Del Valle & Kenning (1985) influence area,

    A_b = min(1, K * N_a * pi * D_d^2 / 4),   K = K_ref * exp(-dT_sub/dT_ref)

The influence area is larger than the bubble footprint (`K > 1`) because the
disturbance extends beyond the bubble itself, and it shrinks with subcooling.

Capping at unity is essential, not cosmetic: with `N_a ~ dT_sup^1.8` the
uncapped expression exceeds one at quite moderate superheat, at which point the
convective term would go negative and the partition would stop making sense.
"""
struct DelValleKenning{F<:AbstractFloat} <: AbstractInfluenceArea
    K_ref::F
    dT_ref::F
end
DelValleKenning(; K_ref=4.8, dT_ref=80.0) = DelValleKenning(float(K_ref), float(dT_ref))
Adapt.@adapt_structure DelValleKenning

@inline function bubble_influence_fraction(
    model::DelValleKenning, s::BoilingState{F}, N_a, D_d) where F
    K = F(model.K_ref)*exp(-max(s.dT_sub, zero(F))/F(model.dT_ref))
    return clamp(K*N_a*F(pi)*D_d^2/4, zero(F), one(F))
end

"""
    ConstantInfluenceArea(; K=2.0)

Fixed influence factor, `A_b = min(1, K * N_a * pi * D_d^2/4)`.

`K = 2` is the original RPI value (Kurul & Podowski). Useful as a control when
assessing how much of a result comes from the subcooling dependence in
[`DelValleKenning`](@ref).
"""
struct ConstantInfluenceArea{F<:AbstractFloat} <: AbstractInfluenceArea
    K::F
end
ConstantInfluenceArea(; K=2.0) = ConstantInfluenceArea(float(K))
Adapt.@adapt_structure ConstantInfluenceArea

@inline bubble_influence_fraction(
    model::ConstantInfluenceArea, s::BoilingState{F}, N_a, D_d) where F =
    clamp(F(model.K)*N_a*F(pi)*D_d^2/4, zero(F), one(F))


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
struct RPI{S,D,Fr,A,P,F<:AbstractFloat} <: AbstractWallBoilingModel
    site_density::S
    departure_diameter::D
    departure_frequency::Fr
    influence_area::A
    patches::P
    Pr_t::F
    alpha_min::F
    wall_capacity::F
    n_iterations::Int
    start_iteration::Int
    friction_velocity::Symbol
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
    wall_capacity = 0.0,
    n_iterations = 40,
    start_iteration = 0,
    friction_velocity = :k)

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
    friction_velocity in (:k, :loglaw) || throw(ArgumentError(
        "`friction_velocity` must be :k or :loglaw, got :$friction_velocity"))

    return RPI(site_density, departure_diameter, departure_frequency, influence_area,
               patches_tuple, float(Pr_t), float(alpha_min), float(wall_capacity),
               n_iterations, start_iteration, friction_velocity)
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

    q_c = h_c*dT_wl*(one(F) - A_b)

    # Quenching: Mikic & Rohsenow transient conduction over the waiting time
    # t_w = 0.8/f. Writing it as sqrt(t_w * k rho cp) keeps the group in terms
    # of the thermal effusivity and avoids dividing by the diffusivity.
    q_q = if f > zero(F)
        t_w = F(0.8)/f
        2*f*sqrt(t_w*s.k_l*s.rho_l*s.cp_l/F(pi))*dT_wl*A_b
    else
        zero(F)
    end

    q_e = N_a*f*(F(pi)/6)*D_d^3*s.rho_v*s.h_fg

    return (q_c=q_c, q_q=q_q, q_e=q_e, A_b=A_b, N_a=N_a, D_d=D_d, f=f)
end

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
        return solve_wall_temperature(rpi, s, q_gen, h_c)

    Cdt = C/F(dt)

    # f increases with T_w. Lower bound: the coldest state in play, where Q -> 0
    # and the storage term is at its most negative. Upper bound: the temperature
    # the wall would reach on storage alone with Q = 0, which cannot be exceeded.
    lo = min(s.T_l, s.T_sat, T_w_prev)
    hi = max(T_w_prev + q_gen/Cdt, lo) + max(q_gen/max(h_c, eps(F)), zero(F))

    for _ in 1:rpi.n_iterations
        mid = (lo + hi)/2
        p = wall_heat_partition(rpi, _at_wall_temperature(s, mid), h_c)
        f = Cdt*(mid - T_w_prev) + (p.q_c + p.q_q + p.q_e) - q_gen
        if f < zero(F)
            lo = mid
        else
            hi = mid
        end
    end

    T_w = (lo + hi)/2
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
@inline function wall_boiling_liquid_factor(rpi::RPI, alpha_l::F) where F
    lo = F(rpi.alpha_min)
    hi = 2*lo
    return clamp((alpha_l - lo)/(hi - lo + eps(F)), zero(F), one(F))
end
