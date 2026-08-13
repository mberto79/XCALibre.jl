# =============================================================================
#  DNB and film boiling - correlation preview
# =============================================================================
#
#  Evaluates the post-CHF closures at the operating point of the boiling curve
#  case, WITHOUT running any CFD. The whole blend is a set of scalar functions of
#  the local fluid state, so where DNB lands and how wide the transition is can
#  be settled in a second rather than in a staircase run.
#
#  Do this BEFORE trusting a film boiling run. The correlations here are
#  property-only by design - that is what makes them transfer to other fluids -
#  but property-only is not the same as right, and hydrogen sits far outside the
#  range most of them were fitted over.
#
#  Two things to read off:
#
#    1. THE CHF COMPARISON. Zuber is a pool boiling limit; flow raises CHF, so
#       Zuber should come in BELOW the measured value. If it comes in above, the
#       correlation is not conservative here and a flow correlation
#       (Katto & Ohno, Shah) should be used through `FixedCriticalHeatFlux`.
#
#    2. dT_CHF VERSUS dT_min. `dT_CHF` comes from inverting the calibrated RPI
#       partition at the CHF flux; `dT_min` from the Leidenfrost closure. If
#       `dT_CHF > dT_min` the two are mutually inconsistent - the nucleate model
#       wants more superheat to reach CHF than the liquid can physically sustain
#       - and the blend interval is being held open by `min_width` alone. That is
#       a statement about the CALIBRATION, not a numerical problem, and it is
#       worth knowing before reading anything into the film branch.
# =============================================================================

using XCALibre, Printf
using XCALibre.ModelPhysics: BoilingState, single_phase_htc, wall_heat_partition,
                             film_closure, film_boiling_fraction

include(joinpath(@__DIR__, "calibrate_rpi_common.jl"))

# Hydrogen critical temperature. The homogeneous nucleation bound needs only
# this one property, which is what makes it portable.
const T_CRIT_H2 = 33.145

# Calibrated nucleate coefficients - keep in step with
# `3d_LH2_pipe_forced_convection.jl`.
const SITE_DENSITY = LemmertChawla(m = 1.105, n = 21.17)
const D_REF        = 1.110e-3

# Measured departure, for comparison only. NOTHING below is fitted to these -
# they are here so the correlations can be scored against them.
const Q_CHF_MEASURED  = 64.0e3      # W/m^2
const DT_CHF_MEASURED = 1.67        # K

state(T_w) = BoilingState(
    T_w = T_w, T_l = T_SAT, T_sat = T_SAT,
    rho_l = LIQ.rho, rho_v = VAP.rho, cp_l = LIQ.cp, k_l = LIQ.k, mu_l = LIQ.mu,
    sigma = SIGMA, h_fg = H_FG, g = 9.81,
    cp_v = VAP.cp, k_v = VAP.k, mu_v = VAP.mu)

const S0 = state(T_SAT)

println("="^78)
@printf("OPERATING POINT   p = %.2f MPa   U = %.2f m/s   D = %.1f mm\n",
        P_SAT/1e6, U_BULK, D_PIPE*1e3)
println("="^78)
@printf("  T_sat  = %8.3f K      h_fg  = %9.1f kJ/kg\n", T_SAT, H_FG/1e3)
@printf("  rho_l  = %8.3f        rho_v = %9.4f  kg/m^3\n", LIQ.rho, VAP.rho)
@printf("  k_v    = %8.5f        mu_v  = %9.3e  cp_v = %.0f J/kg/K\n",
        VAP.k, VAP.mu, VAP.cp)
@printf("  sigma  = %8.3e N/m    Re    = %9.3e   u_tau = %.4f m/s\n",
        SIGMA, RE, U_TAU)
@printf("  h_c    = %8.1f W/m^2/K  (y+ = %.0f)\n", H_C, Y_PLUS)

# -----------------------------------------------------------------------------
# 1. Critical heat flux
# -----------------------------------------------------------------------------
println("\n", "-"^78)
println("CRITICAL HEAT FLUX")
println("-"^78)

rpi_nucleate = RPI(patches = (:w,), site_density = SITE_DENSITY,
                   departure_diameter = TolubinskyKostanchuk(d_ref = D_REF,
                                                             d_max = 4*D_REF))

q_zuber = critical_heat_flux(Zuber(), S0)
q_crowd = critical_heat_flux(BubbleCrowding(), S0, rpi_nucleate, H_C)

@printf("  Zuber (pool, property-only)   %8.2f kW/m^2   %+6.1f%% vs measured\n",
        q_zuber/1e3, 100*(q_zuber/Q_CHF_MEASURED - 1))
@printf("  BubbleCrowding (A_b -> 1)     %8.2f kW/m^2   %+6.1f%% vs measured\n",
        q_crowd/1e3, 100*(q_crowd/Q_CHF_MEASURED - 1))
@printf("  MEASURED (Tatsumoto 2014)     %8.2f kW/m^2\n", Q_CHF_MEASURED/1e3)

println("""
  Zuber is a POOL boiling limit and forced flow raises CHF, so a value below the
  measurement is the expected and conservative outcome. A value above it means
  the pool limit is not bounding this flow and a tube-level correlation should be
  used instead, passed in with `FixedCriticalHeatFlux`.""")

# -----------------------------------------------------------------------------
# 2. Transition endpoints
# -----------------------------------------------------------------------------
println("\n", "-"^78)
println("TRANSITION ENDPOINTS")
println("-"^78)

dT_ber = minimum_film_superheat(Berenson(), S0)
dT_hn  = minimum_film_superheat(HomogeneousNucleation(T_crit = T_CRIT_H2), S0)

T_chf, _ = solve_wall_temperature(rpi_nucleate, S0, Q_CHF_MEASURED, H_C)
dT_chf_model = T_chf - T_SAT

@printf("  dT_CHF  from RPI at measured CHF   %7.3f K   (measured %.2f K)\n",
        dT_chf_model, DT_CHF_MEASURED)
@printf("  dT_min  Berenson                   %7.3f K%s\n", dT_ber,
        T_SAT + dT_ber > T_CRIT_H2 ? "   << UNREACHABLE (past T_crit)" : "")
@printf("  dT_min  homogeneous nucleation     %7.3f K   (0.9 T_crit = %.2f K)\n",
        dT_hn, 0.9*T_CRIT_H2)

if T_SAT + dT_ber > T_CRIT_H2
    println("""
  Berenson returns a wall superheat beyond the critical temperature, i.e. a state
  the liquid cannot occupy. It is a pool boiling correlation whose scales come
  from water-like fluids, and hydrogen's 33 K critical point leaves no room for
  it. This is why `FilmBoiling` accepts a tuple of closures and takes the
  SMALLER - the thermodynamic bound has to be able to override it.""")
end

# -----------------------------------------------------------------------------
# 3. The blended boiling curve
# -----------------------------------------------------------------------------
println("\n", "-"^78)
println("BLENDED BOILING CURVE")
println("-"^78)

rpi = RPI(patches = (:w,), site_density = SITE_DENSITY,
          departure_diameter = TolubinskyKostanchuk(d_ref = D_REF,
                                                    d_max = 4*D_REF),
          wall_capacity = 24.21,
          film_boiling = FilmBoiling(
              chf = FixedCriticalHeatFlux(q = Q_CHF_MEASURED),
              minimum_film = (Berenson(),
                              HomogeneousNucleation(T_crit = T_CRIT_H2)),
              htc = ForcedConvectionFilm()))

fc = film_closure(rpi, rpi.film_boiling, S0, H_C, Y_PLUS, U_TAU)

@printf("\n  blend interval   dT_lo = %.4f K -> dT_hi = %.4f K   (width %.4f K)\n",
        fc.dT_lo, fc.dT_hi, fc.dT_hi - fc.dT_lo)
@printf("  h_film (forced convection in vapour) = %.1f W/m^2/K", fc.h_f)
@printf("   (%.1f%% of h_c)\n", 100*fc.h_f/H_C)

if fc.dT_hi <= fc.dT_lo*(1 + rpi.film_boiling.min_width) + 1e-12
    println("""
  NOTE: the interval is being held open by `min_width`, not by the correlations -
  the Leidenfrost superheat came out at or below the CHF superheat. The blend is
  still well posed (an abrupt DNB), but its width is a numerical choice rather
  than a physical prediction, so do not read the transition shape as physics.""")
end

println("\n   dT_sup [K]    w      q_conv    q_quench   q_evap    q_film    q_total")
println("                        [kW/m^2]  [kW/m^2]  [kW/m^2]  [kW/m^2]  [kW/m^2]")

dTs = sort(unique(vcat(
    collect(range(0.2, fc.dT_lo, length = 6)),
    collect(range(fc.dT_lo, fc.dT_hi, length = 5)),
    fc.dT_hi .* [1.5, 2.0, 3.0, 5.0])))

for dT in dTs
    p = wall_heat_partition(rpi, state(T_SAT + dT), H_C, fc)
    tot = p.q_c + p.q_q + p.q_e + p.q_f
    @printf("  %9.4f  %6.3f  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f\n",
            dT, p.w, p.q_c/1e3, p.q_q/1e3, p.q_e/1e3, p.q_f/1e3, tot/1e3)
end

println("""

  The curve must RISE to CHF, FALL through the transition, then rise again on the
  film branch. That turnover is why a wall thermal capacity is mandatory with
  film boiling: inverting a non-monotone q(T_w) at prescribed flux has up to
  three roots and bisection would pick one of them silently. The transient wall
  balance integrates along the curve instead, so DNB comes out as a fast
  transient rather than a root selection.
""")
