# =============================================================================
#  RPI calibration - LOW flux branch, with the departure diameter fitted
# =============================================================================
#
#  Window: onset up to the knee at ~20 kW/m^2.
#
#  WHY SPLIT THE RANGE. The measured curve is not a power law. Its local exponent
#  is about 1.8 below 20 kW/m^2 and about 11.5 above it, so a single
#  `N_a = (m*dT)^n` cannot represent both - a full-range fit walks to whatever
#  upper bound is placed on `n` and still leaves systematic residuals that pivot
#  about the knee. See `calibrate_rpi_lh2.jl` for that result.
#
#  This half is the well-behaved one: a modest exponent should fit it properly.
#
#  D_REF IS FITTED HERE, and that needs care. `m` and `D_d^3` are multiplicatively
#  degenerate in
#
#      q_evap = N_a * f * (pi/6)*D_d^3 * rho_v * h_fg
#
#  so the two cannot be separated by `q_evap` alone. `D_d` does enter elsewhere -
#  squared in `A_b`, and as 1/sqrt(D_d) in the Cole frequency - which breaks the
#  degeneracy WEAKLY. The grid report prints how much of the parameter space sits
#  within 10% of the optimum: if that fraction is large, the fit is a ridge and
#  the individual numbers should not be quoted.
#
#  The Fritz value for LH2 is ~1110 um, nearly double the 600 um water default,
#  so a fitted `d_ref` far from ~1100 um is itself informative.
# =============================================================================

include(joinpath(@__DIR__, "calibrate_rpi_common.jl"))

const Q_KNEE = 20.0e3     # where the measured curve turns near-vertical

best = run_study(
    "LOW FLUX  (onset -> knee), fitting m, n and d_ref",
    (m, n) -> LemmertChawla(m = m, n = n),
    (10.0 .^ range(-1.0, 3.0, length = 40),        # m
     range(1.0, 8.0, length = 40),                 # n
     range(200e-6, 3000e-6, length = 30));         # d_ref [m]
    q_min = 0.0, q_max = Q_KNEE, fit_d_ref = true)

println("""

USE
    site_density       = LemmertChawla(m = $(round(best.p[1], sigdigits=5)), n = $(round(best.p[2], sigdigits=5))),
    departure_diameter = TolubinskyKostanchuk(d_ref = $(round(best.p[3], sigdigits=4)),
                                              d_max = $(round(4*best.p[3], sigdigits=4))),

VALID ONLY for q_w <= $(Q_KNEE/1e3) kW/m^2 at $(P_SAT/1e6) MPa and $(U_BULK) m/s.
Above the knee use the high-flux fit; the two are not interchangeable and there
is no guarantee they meet smoothly at the crossover - check that they do before
using them together.
""")
