export AbstractPhaseChangeModel, AbstractSaturationModel
export Lee, Schrage, ModifiedEnergyJump
export Antoine, saturation_pressure, saturation_temperature
export phase_change_rate!

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
struct Schrage{T<:AbstractFloat} <: AbstractPhaseChangeModel
    sigma::T
end
Schrage(; sigma=1.0e-3) = Schrage(float(sigma))

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
struct Lee{T<:AbstractFloat} <: AbstractPhaseChangeModel
    sigma::T
end
Lee(; sigma=1.0e-6) = Lee(float(sigma))


# =============================================================================
#  Rate evaluation
# =============================================================================

"""
    phase_change_rate!(mdot, pc, alpha, gradAlphaMag, T, p_abs,
                       rho_l, rho_v, sat, L, R_sp, config)

Fill `mdot` with the **volumetric** phase change rate [kg/m^3/s], positive for
evaporation:

    mdot = mdot''(model) * |grad(alpha)|

`pc === nothing` zeroes the field, which is the no-phase-change case.
"""
function phase_change_rate!(mdot, ::Nothing, alpha, gradAlphaMag, T, p_abs,
                            rho_l, rho_v, sat, L, R_sp, config)
    fill!(mdot.values, zero(eltype(mdot.values)))
    return nothing
end

function phase_change_rate!(mdot, pc::AbstractPhaseChangeModel, alpha, gradAlphaMag,
                            T, p_abs, rho_l, rho_v, sat, L, R_sp, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(mdot)
    kernel! = _phase_change_rate!(_setup(backend, workgroup, ndrange)...)
    kernel!(mdot, pc, alpha, gradAlphaMag, T, p_abs, rho_l, rho_v, sat, L, R_sp)
    return nothing
end

@kernel inbounds=true function _phase_change_rate!(
    mdot, pc, alpha, gradAlphaMag, T, p_abs, rho_l, rho_v, sat, L, R_sp)
    i = @index(Global)
    t = T[i]
    p = p_abs[i]
    T_sat = saturation_temperature(sat, p)
    flux = interfacial_mass_flux(pc, alpha[i], t, p, T_sat,
                                 rho_l[i], rho_v[i], sat, L, R_sp)
    mdot[i] = flux*gradAlphaMag[i]
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
