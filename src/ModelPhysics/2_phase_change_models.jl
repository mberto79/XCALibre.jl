export AbstractPhaseChangeModel, AbstractSaturationModel
export Lee, Schrage, ModifiedEnergyJump
export Antoine, saturation_pressure, saturation_temperature
export phase_change_rate!
export AbstractInterfacialArea, ResolvedInterface, DispersedBubbles
export interfacial_area_density, AIAD, aiad_weights
export LAST_UR

"""
    LAST_UR[]

The slip velocity field `Ur` from the most recent multiphase step, or `nothing`.

DIAGNOSTIC ONLY - nothing in the solver reads it back. It exists because `Ur` is
local to the multiphase solver while `save_output` lives here, so there is
otherwise no way to get the slip into the output for inspection.

Why it is needed: on the LH2 pipe the near-wall void refuses to rise past ~0.5
even with wall evaporation running at full rate (`K_dry = 0`), so vapour is being
REMOVED as fast as it is made. Which term does the removing - drift, lift, wall
lubrication, turbulent dispersion - cannot be told apart from `alpha` and `U`
alone, because lift and wall lubrication both act by modifying `Ur`. Writing `Ur`
makes the near-wall vapour budget measurable instead of inferred.

Set once per step by the multiphase solver; read by `save_output`. Follows the
same stash pattern as `LAST_WALL_REPORT`, in the opposite direction, because
`ModelPhysics` is loaded BEFORE `Solvers`.
"""
const LAST_UR = Ref{Any}(nothing)


"""
    AbstractPhaseChangeModel

Supertype for interfacial liquid–vapour phase change models.

Every model returns an interfacial **mass flux** `mdot''` [kg/m^2/s], positive for
evaporation. The solver converts that to a volumetric source by multiplying by the
interfacial area density `|grad(alpha)|` [1/m] (Fernandes et al. Eq. 6):

    S_alpha = mdot'' * |grad(alpha)|          [kg/m^3/s]

which then feeds three places:

  - volume fraction:  `-S_alpha/rho_l`
  - pressure:         `+S_alpha*(1/rho_v - 1/rho_l)`   (net volume creation)
  - energy:           `-S_alpha*L`                     (latent heat)
"""
abstract type AbstractPhaseChangeModel end

"""
    AbstractSaturationModel

Supertype for saturation pressure/temperature relations.
"""
abstract type AbstractSaturationModel end


# =============================================================================
#  Saturation relation
# =============================================================================

"""
    Antoine(; A, B, C, Tmin=-Inf, Tmax=Inf)

Antoine saturation curve,

    log10(p_sat [bar]) = A - B/(T_sat + C)

`saturation_pressure` returns Pa; `saturation_temperature` inverts the same
expression analytically.

The default coefficients are the empirical fit to NIST hydrogen data given by
Fernandes et al. Eq. (15): `A = 3.54314`, `B = 99.395`, `C = 7.726`, stated valid
over 21.01–32.27 K.

Note that the K-Site operating point (103 kPa, T_sat ~ 20.43 K) sits just *below*
that stated range, so the paper is itself extrapolating slightly. `Tmin`/`Tmax`
are carried for reference and are not enforced — clamping would distort the
saturation state rather than improve it.

### Example
    Antoine()                                  # hydrogen, paper Eq. (15)
    Antoine(A=3.54314, B=99.395, C=7.726)      # explicit
"""
struct Antoine{T<:AbstractFloat} <: AbstractSaturationModel
    A::T
    B::T
    C::T
    Tmin::T
    Tmax::T
end

Antoine(; A=3.54314, B=99.395, C=7.726, Tmin=21.01, Tmax=32.27) =
    Antoine(float(A), float(B), float(C), float(Tmin), float(Tmax))

const _BAR_TO_PA = 1.0e5

"""
    saturation_pressure(sat::Antoine, T) -> Pa

Saturation pressure at temperature `T`.
"""
@inline function saturation_pressure(sat::Antoine, T)
    return _BAR_TO_PA*exp10(sat.A - sat.B/(T + sat.C))
end

"""
    saturation_temperature(sat::Antoine, p) -> K

Saturation temperature at pressure `p` [Pa]; the analytic inverse of
[`saturation_pressure`](@ref).
"""
@inline function saturation_temperature(sat::Antoine, p)
    return sat.B/(sat.A - log10(p/_BAR_TO_PA)) - sat.C
end


# =============================================================================
#  Phase change models
# =============================================================================

"""
    Schrage(; sigma=1.0e-3)

Hertz–Knudsen–Schrage kinetic model (Fernandes et al. Eq. 14), in the
near-equilibrium form obtained by taking `p_l = p_sat(T_sat)` and
`T_l = T_v = T_sat`:

    mdot'' = (2*sigma/(2 - sigma)) * sqrt(1/(2*pi*R_sp*T_sat)) * (p_sat - p_v)

`sigma` is the dimensionless accommodation coefficient and `R_sp` the specific gas
constant of the vapour (so `sqrt(M/(2*pi*R_u*T)) == sqrt(1/(2*pi*R_sp*T))`).

The paper finds this the most accurate and least coefficient-sensitive of the
three (<= 3.0 % MAPE), because the mass flux self-regulates: raising `sigma`
injects more vapour, which raises `p_v` and shrinks the driving `(p_sat - p_v)`.
Values above 1e-2 require time steps of order 1e-3 s for stability; 1e-3 is
recommended.

Parametric values used in the paper: 1e-3 (baseline), 1e-4, 1e-5.
"""
struct Schrage{T<:AbstractFloat,R} <: AbstractPhaseChangeModel
    sigma::T
    R::R
end

# `R` is an OPTIONAL override for the vapour specific gas constant [J/kg/K],
# used only when the vapour equation of state cannot supply one (`ConstEos`).
# The kinetic prefactor sqrt(1/(2 pi R_sp T_sat)) is a property of the SUBSTANCE,
# not of the equation of state, so a constant-density vapour has a perfectly
# well-defined value for it - there is simply nowhere on `ConstEos` to keep it.
Schrage(; sigma=1.0e-3, R=nothing) = Schrage(float(sigma), R === nothing ? nothing : float(R))

"""
    ModifiedEnergyJump(; h)

Modified Energy Jump model (Fernandes et al. Eq. 9), an energy-transport model
based on the interfacial energy balance:

    mdot'' = h*(T - T_sat)/L

`h` is the liquid–vapour heat transfer coefficient [W/m^2/K] and `L` the latent
heat [J/kg]. There is no kinetic term, so once `h` is prescribed it acts as a
lumped parameter controlling the whole phase change rate — which is why the paper
finds this model accurate but strongly `h`-sensitive.

`h` may be estimated from `h = k*C*lambda/l * Ra^n` with `C = 0.27`, `n = 0.25`
and `k` a calibration coefficient (paper Eq. 10, after Matveev); for liquid
hydrogen it typically falls in 1–10 W/m^2/K.

Parametric values used in the paper: 1.0 (baseline), 10.0, 100.0. For the K-Site
tank ~10 W/m^2/K reproduced the non-linear pressure profile best.
"""
struct ModifiedEnergyJump{T<:AbstractFloat} <: AbstractPhaseChangeModel
    h::T
end
ModifiedEnergyJump(; h) = ModifiedEnergyJump(float(h))

"""
    Lee(; r)

Lee (1980) relaxation model, in the standard volumetric form:

    mdot = r*alpha_l*rho_l*(T - T_sat)/T_sat     for T > T_sat  (evaporation)
    mdot = r*alpha_v*rho_v*(T - T_sat)/T_sat     for T < T_sat  (condensation)

`r` is the relaxation coefficient [1/s], prescribed directly, and the SAME value
is used for both branches. `mdot` is a volumetric mass source [kg/m^3/s].

Note the sign convention: `(T - T_sat)/T_sat` carries the sign in both branches,
so only the `alpha*rho` weighting switches phase. The branches are not symmetric
even at a single `r`: their ratio is `(alpha_l*rho_l)/(alpha_v*rho_v)`, which for
LH2/GH2 at 0.4 MPa and 10% void is about 117, so condensation is far slower than
evaporation at equal departure from saturation. That is the model, not a defect.

# `alpha_l` and `alpha_v` mean the LIQUID and VAPOUR fractions

Not the tracked fraction. The two coincide only when `alpha` happens to track the
liquid; when it tracks the vapour - which is what `multiphase_liquid_phase`
recommends, and what the LH2 pipe case does - they are swapped. The caller is
responsible for passing the LIQUID fraction, and `phase_change_rate!` does.

Getting this wrong is not a small error: measured on rung 3.1, the identical
physical state gave a relaxation time of 25.0 s against 4.02 s, a factor of 6.2.

# No interfacial area factor

`Schrage` and `ModifiedEnergyJump` return a mass flux PER UNIT INTERFACE AREA and
are multiplied by `a_i` to become volumetric. Lee's `r` is already volumetric, so
it is not - see [`uses_interfacial_area`](@ref). Multiplying it by `a_i` would
make the effective coefficient proportional to `alpha*(1 - alpha)`, so it would
vanish in a nearly pure cell regardless of superheat, and `r` would no longer be
the coefficient any published Lee calibration refers to.

# Choosing `r`

There is no universal value; it is a numerical relaxation rate, and the usual
guidance is to make it large enough that the interface stays near saturation
without making the source stiff. Commercial defaults are O(0.1-100) 1/s. The
verification statement is that as `r` grows the solution must converge onto the
thermally-limited (Stefan) answer - see rung 3.2 of the validation plan.
"""
struct Lee{T<:AbstractFloat} <: AbstractPhaseChangeModel
    r::T
end

function Lee(; r=nothing, sigma=nothing, R=nothing)
    sigma === nothing || throw(ArgumentError(
        """`Lee(sigma = ...)` has been removed.

`sigma` was an accommodation coefficient from which the relaxation parameter was
DERIVED, as

    beta = sigma*sqrt(1/(2*pi*R_sp*T_sat))*L*rho_l/(rho_l - rho_v)

so that Lee and `Schrage` could be driven from the same knob. It is not the `r`
of the published Lee model, it carried different units, and it was additionally
scaled by the interfacial area density.

Pass the relaxation coefficient directly instead:

    Lee(r = 100.0)      # [1/s]

There is no exact conversion: the old form was area-scaled and state-dependent.
As a rough guide at alpha = 0.9 for LH2/GH2 at 0.4 MPa with d = 1 mm, the old
`sigma = 1e-6` corresponded to an effective r of about 0.29 1/s."""))
    R === nothing || throw(ArgumentError(
        """`Lee(R = ...)` has been removed along with `sigma`.

`R` supplied the vapour specific gas constant for the kinetic prefactor
`sqrt(1/(2*pi*R_sp*T_sat))`. The prescribed-coefficient form has no kinetic
prefactor, so it needs no gas constant and works with any equation of state,
including `ConstEos`."""))
    r === nothing && throw(ArgumentError(
        "`Lee` needs its relaxation coefficient: `Lee(r = 100.0)`  # [1/s]"))
    r > 0 || throw(ArgumentError("`Lee` needs a positive `r` [1/s], got $r"))
    return Lee(float(r))
end


# =============================================================================
#  Rate evaluation
# =============================================================================

# =============================================================================
#  Interfacial area density
# =============================================================================

"""
    AbstractInterfacialArea

How the interfacial area per unit volume `a_i` [1/m] is closed. The phase change
models return a mass flux PER UNIT INTERFACE AREA, so `a_i` is what converts
that into the volumetric rate the equations need:

    mdot = mdot''(model) * a_i

The right closure depends on whether the interface is RESOLVED or DISPERSED, and
getting it wrong is not a small error - see [`ResolvedInterface`](@ref).
"""
abstract type AbstractInterfacialArea end

"""
    interfacial_area_density(model, alpha, gradAlphaMag) -> a_i [1/m]
"""
function interfacial_area_density end

"""
    ResolvedInterface()

`a_i = |grad(alpha)|`. The VOF closure, and correct there: for an interface
smeared over a couple of cells, integrating `|grad(alpha)|` through the interface
region recovers its area exactly.

### Do not use it for a dispersed flow

`|grad(alpha)|` is not an area density in any general sense - it is a statement
that all the interface is at the place where `alpha` changes. In a dispersed
bubbly mixture there is no resolved interface, `alpha` varies smoothly, and the
cell-to-cell variation the gradient measures is NUMERICAL rather than physical.

That makes it a feedback path with a checkerboard eigenmode: a 2*dx oscillation
in `alpha` is precisely the field that MAXIMISES `|grad(alpha)|` for a given
amplitude, so the phase change source is largest exactly where the solution is
least physical, and grows as the oscillation grows. Measured on the LH2 pipe at
0.4 MPa with 86 um wall cells: the physical dispersed value is 24 1/m, while a
checkerboard of amplitude 0.1 gives 1165 1/m - a factor of 49, and one that
increases with the noise it is responding to.

It is also wrong in sign of behaviour at the boundaries of the flow: it is
LARGEST at a pure-liquid cell adjacent to a bubbly one, where there is least
interface, and zero in a uniformly bubbly region, where there is most.
"""
struct ResolvedInterface <: AbstractInterfacialArea end
Adapt.@adapt_structure ResolvedInterface

@inline interfacial_area_density(::ResolvedInterface, alpha, gradAlphaMag) =
    gradAlphaMag

"""
    DispersedBubbles(; diameter)

`a_i = 6 alpha_d (1 - alpha_d) / d`, the interfacial area density of a dispersed
phase of spherical inclusions of diameter `d` at volume fraction `alpha_d`.

The strict result for spheres is `6 alpha_d/d`; the extra `(1 - alpha_d)` makes
the expression symmetric under phase inversion so it vanishes at BOTH limits
rather than growing without bound as the dispersed phase takes over. It is the
form mixture models normally carry, and being symmetric it does not matter which
phase `alpha` tracks.

### Why this and not `|grad(alpha)|`

It is bounded, smooth, and a purely LOCAL algebraic function of `alpha` - there
is no gradient in it, so it cannot amplify a cell-to-cell oscillation the way
[`ResolvedInterface`](@ref) does.

It is also the CONSISTENT choice for a drift-flux mixture. That model already
closes the slip velocity through a per-bubble force balance with
`tau_d = rho_d d^2/(18 mu_c)`, i.e. it has already committed to a bubble
diameter. Using `|grad(alpha)|` for mass transfer while using `d` for momentum
makes the two closures describe different dispersed phases. Passing the same `d`
to both removes that inconsistency and introduces no new free parameter.
"""
struct DispersedBubbles{F<:AbstractFloat} <: AbstractInterfacialArea
    diameter::F
end
function DispersedBubbles(; diameter)
    diameter > 0 || throw(ArgumentError(
        "`diameter` must be a positive bubble diameter [m], got $diameter"))
    return DispersedBubbles(float(diameter))
end
Adapt.@adapt_structure DispersedBubbles

@inline function interfacial_area_density(
    model::DispersedBubbles, alpha::F, gradAlphaMag) where F
    a = clamp(alpha, zero(F), one(F))
    return 6*a*(one(F) - a)/F(model.diameter)
end

"""
    AIAD(; d_bubble, d_droplet, alpha_bubbly=0.3, alpha_droplet=0.3, sharpness=70.0)

Algebraic Interfacial Area Density (Hoehne & Vallee, 2010). Blends THREE
morphologies on the local volume fraction instead of committing to one:

    f_B  = 1/(1 + exp(s*(alpha_g - alpha_bubbly)))     gas dispersed in liquid
    f_D  = 1/(1 + exp(s*(alpha_l - alpha_droplet)))    liquid dispersed in gas
    f_FS = 1 - f_B - f_D                               resolved free surface

    a_i  = f_B*6*alpha_g/d_bubble + f_FS*|grad(alpha)| + f_D*6*alpha_l/d_droplet

### Why blend at all

`DispersedBubbles` assumes gas inclusions in liquid at every void fraction, and
`ResolvedInterface` assumes a resolved interface everywhere. Neither survives a
flow that starts bubbly at the inlet and reaches 35-45% void by CHF: the first
has no notion of the liquid becoming dispersed, the second no notion of there
being no interface to resolve.

### CONVENTION - this model is NOT symmetric

`interfacial_area_density` is called with the LIQUID fraction, and the two
existing closures do not care because `6a(1-a)/d` and `|grad(alpha)|` are both
unchanged by `a -> 1-a`. AIAD is different - which phase is dispersed is the
whole point - so it reads its argument as `alpha_l` and forms
`alpha_g = 1 - alpha_l`. Passing the gas fraction instead inverts the morphology
map.

### WARNING - the free-surface branch reintroduces a gradient feedback

`f_FS*|grad(alpha)|` is active only in the mid-void band (roughly
0.3 < alpha_g < 0.7), but inside it the objection under `ResolvedInterface`
applies in full: a 2*dx checkerboard is the field that MAXIMISES
`|grad(alpha)|`, so the phase-change source is largest exactly where the solution
is least physical. Measured on this case, a checkerboard of amplitude 0.1 gave
1165 1/m against a physical dispersed value of 24 1/m. The blend confines that to
the band where a resolved interface is genuinely the right picture; it does not
remove it. Watch the mid-void region for the same signature.

`sharpness = 70` and the 0.3 limits are the published values.
"""
struct AIAD{F<:AbstractFloat} <: AbstractInterfacialArea
    d_bubble::F
    d_droplet::F
    alpha_bubbly::F
    alpha_droplet::F
    sharpness::F
end
function AIAD(; d_bubble, d_droplet, alpha_bubbly = 0.3, alpha_droplet = 0.3,
                sharpness = 70.0)
    d_bubble > 0 || throw(ArgumentError("`d_bubble` must be positive, got $d_bubble"))
    d_droplet > 0 || throw(ArgumentError("`d_droplet` must be positive, got $d_droplet"))
    0 < alpha_bubbly < 1 || throw(ArgumentError(
        "`alpha_bubbly` must be in (0,1), got $alpha_bubbly"))
    0 < alpha_droplet < 1 || throw(ArgumentError(
        "`alpha_droplet` must be in (0,1), got $alpha_droplet"))
    sharpness > 0 || throw(ArgumentError("`sharpness` must be positive, got $sharpness"))
    return AIAD(float(d_bubble), float(d_droplet), float(alpha_bubbly),
                float(alpha_droplet), float(sharpness))
end
Adapt.@adapt_structure AIAD

"""
    aiad_weights(model, alpha_l) -> (f_bubble, f_freesurface, f_droplet)

Morphology weights, summing to 1. `f_FS` is the remainder, floored at zero - which
only binds if the two limits are set close enough for the sigmoids to overlap.
"""
@inline function aiad_weights(m::AIAD, alpha_l::F) where F
    a_l = clamp(alpha_l, zero(F), one(F))
    a_g = one(F) - a_l
    fB = one(F)/(one(F) + exp(F(m.sharpness)*(a_g - F(m.alpha_bubbly))))
    fD = one(F)/(one(F) + exp(F(m.sharpness)*(a_l - F(m.alpha_droplet))))
    fFS = max(zero(F), one(F) - fB - fD)
    return (fB, fFS, fD)
end

@inline function interfacial_area_density(m::AIAD, alpha_l::F, gradAlphaMag) where F
    a_l = clamp(alpha_l, zero(F), one(F))
    a_g = one(F) - a_l
    fB, fFS, fD = aiad_weights(m, a_l)
    return fB*6*a_g/F(m.d_bubble) + fFS*gradAlphaMag + fD*6*a_l/F(m.d_droplet)
end


# =============================================================================
#  Rate evaluation
# =============================================================================

"""
    phase_change_rate!(mdot, pc, area, alpha, gradAlphaMag, T, p_abs,
                       rho_l, rho_v, sat, L, R_sp, config)

Fill `mdot` with the **volumetric** phase change rate [kg/m^3/s], positive for
evaporation:

    mdot = mdot''(model) * a_i(area, alpha, |grad(alpha)|)

`pc === nothing` zeroes the field, which is the no-phase-change case. `area`
selects the interfacial area closure - see [`AbstractInterfacialArea`](@ref).
"""
function phase_change_rate!(mdot, ::Nothing, area, alpha, gradAlphaMag, T, p_abs,
                            rho_l, rho_v, sat, L, R_sp, config)
    fill!(mdot.values, zero(eltype(mdot.values)))
    return nothing
end

function phase_change_rate!(mdot, pc::AbstractPhaseChangeModel, area, alpha,
                            gradAlphaMag, T, p_abs, rho_l, rho_v, sat, L, R_sp,
                            config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(mdot)
    kernel! = _phase_change_rate!(_setup(backend, workgroup, ndrange)...)
    kernel!(mdot, pc, area, alpha, gradAlphaMag, T, p_abs, rho_l, rho_v, sat, L, R_sp,
            Val(uses_interfacial_area(pc)))
    return nothing
end

"""
    uses_interfacial_area(model) -> Bool

Whether the model returns a mass flux PER UNIT INTERFACE AREA [kg/m^2/s], which
must then be multiplied by `a_i` [1/m] to become the volumetric source the
equations want.

`Schrage` and `ModifiedEnergyJump` do: the first is a kinetic-theory flux across
an interface, the second is `h*(T - T_sat)/L` with `h` in W/m^2/K. Both are
meaningless without an interface area.

`Lee` does NOT. Its `r` is already a volumetric relaxation rate [1/s], so
multiplying by `a_i` would make the effective coefficient proportional to
`alpha*(1 - alpha)` - vanishing in a nearly pure cell whatever the superheat -
and `r` would no longer be the quantity any published Lee calibration refers to.
"""
uses_interfacial_area(::AbstractPhaseChangeModel) = true
uses_interfacial_area(::Lee) = false

@kernel inbounds=true function _phase_change_rate!(
    mdot, pc, area, alpha_l, gradAlphaMag, T, p_abs, rho_l, rho_v, sat, L, R_sp,
    area_scaled)
    i = @index(Global)
    t = T[i]
    p = p_abs[i]
    a_l = alpha_l[i]
    T_sat = saturation_temperature(sat, p)
    flux = interfacial_mass_flux(pc, a_l, t, p, T_sat,
                                 rho_l[i], rho_v[i], sat, L, R_sp)
    # `Val` so the branch resolves at compile time and the kernel stays GPU-safe.
    mdot[i] = _scale_by_area(area_scaled, flux, area, a_l, gradAlphaMag[i])
end

# `a_i = 6*a*(1 - a)/d` is symmetric under `a -> 1 - a`, so the dispersed closure
# does not care which phase the fraction measures. `ResolvedInterface` reads
# |grad(alpha)|, which is likewise unchanged by the swap.
@inline _scale_by_area(::Val{true}, flux, area, a, g) =
    flux*interfacial_area_density(area, a, g)
@inline _scale_by_area(::Val{false}, flux, area, a, g) = flux

"""
    _kinetic_prefactor(T_sat, R_sp)

`sqrt(1/(2*pi*R_sp*T_sat))`, the kinetic-theory group shared by the Schrage and
Lee models. Equal to `sqrt(M/(2*pi*R_u*T_sat))` since `R_sp = R_u/M`.
"""
@inline _kinetic_prefactor(T_sat, R_sp) = sqrt(one(T_sat)/(2*pi*R_sp*T_sat))

"""
    interfacial_mass_flux(model, alpha, T, p, T_sat, rho_l, rho_v, sat, L, R_sp)

Interfacial mass flux [kg/m^2/s], positive for evaporation.
"""
@inline function interfacial_mass_flux(
    pc::Schrage, alpha, T, p, T_sat, rho_l, rho_v, sat, L, R_sp)
    s = pc.sigma
    p_sat = saturation_pressure(sat, T)
    return (2*s/(2 - s))*_kinetic_prefactor(T_sat, R_sp)*(p_sat - p)
end

@inline function interfacial_mass_flux(
    pc::ModifiedEnergyJump, alpha, T, p, T_sat, rho_l, rho_v, sat, L, R_sp)
    return pc.h*(T - T_sat)/L
end

@inline function interfacial_mass_flux(
    pc::Lee, alpha_l, T, p, T_sat, rho_l, rho_v, sat, L, R_sp)
    # `alpha_l` is the LIQUID fraction, not the tracked one - see the docstring.
    # (T - T_sat)/T_sat carries the sign; only the alpha*rho weighting switches.
    driving = (T - T_sat)/T_sat
    weight = T > T_sat ? alpha_l*rho_l : (one(alpha_l) - alpha_l)*rho_v
    return pc.r*weight*driving
end
