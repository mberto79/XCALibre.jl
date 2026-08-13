# =============================================================================
#  RPI calibration - Hibiki & Ishii site density
# =============================================================================
#
#  Tests whether a DIFFERENT FUNCTIONAL FORM can do what a power law cannot.
#
#      N_a = N_bar*(1 - exp(-theta^2/(8 mu_c^2)))*(exp(f(rho+)*lambda/R_c) - 1)
#
#  with the critical cavity radius from Clausius-Clapeyron, so `N_a` depends on
#  superheat through `R_c ~ 1/dT_sup` inside an EXPONENTIAL rather than a power.
#  That is a much steeper dependence, which is the property the near-vertical
#  high-flux branch needs.
#
#  EXPECT IT TO SATURATE. `HibikiIshii`'s own docstring records that for LH2 the
#  `N_max` cap "binds almost immediately", making the model effectively constant.
#  A constant site density gives `q_evap` no superheat dependence at all, which is
#  the opposite failure to the power law: instead of being too shallow it becomes
#  flat, and the partition then closes almost entirely through convection and
#  quenching.
#
#  This study is therefore worth running to CONFIRM or REFUTE that note against
#  real data rather than to produce coefficients. Watch `N_max`: if the best fit
#  sits against it, the model is saturated and the result says nothing about
#  `N_bar` or `lambda`.
#
#  `lambda` and `N_bar` are swept because they are the two coefficients with a
#  physical scale here - `lambda` is a cavity length scale, `N_bar` the amplitude.
#  `d_ref` is held at the Fritz value so the comparison against the LemmertChawla
#  studies isolates the site-density form.
# =============================================================================

include(joinpath(@__DIR__, "calibrate_rpi_common.jl"))

# N_max is raised well above the default so the cap does NOT silently bind. If
# the fit still wants to sit against it, that is a real result about the model.
const N_MAX_TEST = 1.0e16

best = run_study(
    "HIBIKI-ISHII site density, full nucleate branch",
    (N_bar, lambda) -> HibikiIshii(N_bar = N_bar, lambda = lambda,
                                   N_max = N_MAX_TEST),
    (10.0 .^ range(2.0, 9.0, length = 45),         # N_bar [1/m^2]
     10.0 .^ range(-8.0, -4.0, length = 45));      # lambda [m]
    q_min = 0.0, q_max = Q_CHF, fit_d_ref = false, d_ref_fixed = D_FRITZ)

# Saturation shows up in the FIT itself: if `N_a` were pinned at the cap the
# predicted superheat would lose its flux dependence and the residuals would be
# flat rather than sloped. Reading it off the fit avoids reaching into
# `nucleation_site_density`, which is internal and not part of the public API.

println("""

COMPARE against the LemmertChawla studies by RMS(log dT) over the SAME window
(`calibrate_rpi_lh2.jl` reports the full-range figure). A lower RMS here would
mean the exponential form genuinely fits the near-vertical branch better; a
similar or worse one means the site-density form is not what is limiting the fit.
""")
