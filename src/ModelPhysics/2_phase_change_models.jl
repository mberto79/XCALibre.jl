export AbstractPhaseChangeModel, AbstractSaturationModel
export Lee, Schrage, ModifiedEnergyJump
export Antoine, saturation_pressure, saturation_temperature
export phase_change_rate!
export AbstractInterfacialArea, ResolvedInterface, DispersedBubbles
export interfacial_area_density

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
    Lee(; sigma=1.0e-6)

Lee (1980) relaxation model (Fernandes et al. Eqs. 11–12):

    mdot'' = beta*alpha_l*rho_l*(T - T_sat)/T_sat     for T > T_sat  (evaporation)
    mdot'' = beta*alpha_v*rho_v*(T - T_sat)/T_sat     for T < T_sat  (condensation)

Rather than prescribing the relaxation parameter `beta` directly (the usual
commercial implementation), it is derived from an accommodation coefficient so
that Lee and [`Schrage`](@ref) can be compared on the same footing:

    beta = sigma * sqrt(1/(2*pi*R_sp*T_sat)) * L*rho_l/(rho_l - rho_v)

Note the sign convention: `(T - T_sat)/T_sat` carries the sign in both branches,
so only the `alpha*rho` weighting switches phase.

The paper finds this the least accurate of the three (up to 11 % MAPE) and
reports `sigma = 1e-6` giving non-physical, diverging boil-off. That behaviour is
a property of the model, not a defect to fix — reproducing it is part of
reproducing the paper.

Parametric values used in the paper: 1e-6 (baseline), 1e-7, 1e-8.
"""
struct Lee{T<:AbstractFloat,R} <: AbstractPhaseChangeModel
    sigma::T
    R::R
end

# `R` is an OPTIONAL override for the vapour specific gas constant [J/kg/K],
# used only when the vapour equation of state cannot supply one (`ConstEos`).
# The kinetic prefactor sqrt(1/(2 pi R_sp T_sat)) is a property of the SUBSTANCE,
# not of the equation of state, so a constant-density vapour has a perfectly
# well-defined value for it - there is simply nowhere on `ConstEos` to keep it.
Lee(; sigma=1.0e-6, R=nothing) = Lee(float(sigma), R === nothing ? nothing : float(R))


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
    kernel!(mdot, pc, area, alpha, gradAlphaMag, T, p_abs, rho_l, rho_v, sat, L, R_sp)
    return nothing
end

@kernel inbounds=true function _phase_change_rate!(
    mdot, pc, area, alpha, gradAlphaMag, T, p_abs, rho_l, rho_v, sat, L, R_sp)
    i = @index(Global)
    t = T[i]
    p = p_abs[i]
    T_sat = saturation_temperature(sat, p)
    flux = interfacial_mass_flux(pc, alpha[i], t, p, T_sat,
                                 rho_l[i], rho_v[i], sat, L, R_sp)
    mdot[i] = flux*interfacial_area_density(area, alpha[i], gradAlphaMag[i])
end

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
    pc::Lee, alpha, T, p, T_sat, rho_l, rho_v, sat, L, R_sp)
    # beta from the accommodation coefficient (paper Eq. 12)
    drho = rho_l - rho_v
    beta = pc.sigma*_kinetic_prefactor(T_sat, R_sp)*L*rho_l/drho

    # (T - T_sat)/T_sat carries the sign; only the alpha*rho weighting switches.
    driving = (T - T_sat)/T_sat
    weight = T > T_sat ? alpha*rho_l : (one(alpha) - alpha)*rho_v
    return beta*weight*driving
end
