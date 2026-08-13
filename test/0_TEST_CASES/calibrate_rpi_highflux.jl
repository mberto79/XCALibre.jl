# =============================================================================
#  RPI calibration - HIGH flux branch, with the departure diameter fitted
# =============================================================================
#
#  Window: the knee at ~20 kW/m^2 up to CHF at 64 kW/m^2.
#
#  THIS IS THE HARD HALF. Over this window the measured superheat barely moves -
#  1.53 to 1.64 K while the flux triples - so the branch is very nearly VERTICAL
#  and its local exponent is around 11.5.
#
#  `q_total = q_conv + q_quench + q_evap` has a FINITE slope in `dT_sup` for any
#  finite `n`, so a vertical branch is not reachable: the fit will push `n` to
#  whatever bound it is given. If the report flags the optimum on a grid edge,
#  that is this limitation showing, not a bad grid - widening it will simply move
#  the answer to the new edge.
#
#  What a fit here is therefore worth: it gives the best available representation
#  over the window, and the residual tells you how much error a single power law
#  costs in the regime that matters most for approaching CHF. It should NOT be
#  read as a physical site density.
#
#  A near-vertical branch also means CHF prediction is intrinsically brittle - a
#  small superheat error maps to an enormous flux error - which is worth knowing
#  before relying on any extrapolation towards departure.
# =============================================================================

include(joinpath(@__DIR__, "calibrate_rpi_common.jl"))

const Q_KNEE = 20.0e3

best = run_study(
    "HIGH FLUX  (knee -> CHF), fitting m, n and d_ref",
    (m, n) -> LemmertChawla(m = m, n = n),
    (10.0 .^ range(-2.0, 3.0, length = 40),        # m
     range(1.0, 20.0, length = 40),                # n - deliberately wide
     range(200e-6, 3000e-6, length = 30));         # d_ref [m]
    q_min = Q_KNEE, q_max = Q_CHF, fit_d_ref = true)

println("""

USE
    site_density       = LemmertChawla(m = $(round(best.p[1], sigdigits=5)), n = $(round(best.p[2], sigdigits=5))),
    departure_diameter = TolubinskyKostanchuk(d_ref = $(round(best.p[3], sigdigits=4)),
                                              d_max = $(round(4*best.p[3], sigdigits=4))),

VALID ONLY for $(Q_KNEE/1e3) <= q_w <= $(Q_CHF/1e3) kW/m^2 at $(P_SAT/1e6) MPa and $(U_BULK) m/s.

If `n` came back on the grid edge, the power-law form is the binding constraint
rather than the coefficients, and no widening will fix it. In that case the
honest options are to accept the residual over this window, or to change the
site-density model - see `calibrate_rpi_hibiki.jl`.
""")
