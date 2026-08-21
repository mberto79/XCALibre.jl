# =============================================================================
#  Standalone calibration of the RPI wall boiling closures against LH2 data
# =============================================================================
#
#  NO CFD. The RPI partition is local and algebraic: given (q_w, T_l, h_c, fluid
#  properties) `solve_wall_temperature` returns the wall temperature directly. So
#  a candidate parameter set costs microseconds to evaluate rather than an
#  overnight 3D run, and the whole parameter space can be swept in seconds.
#
#  The only CFD-derived input is `h_c`, and for saturated boiling `T_l = T_sat`,
#  so there is nothing else to couple. Verify the fitted closure in one or two
#  full runs afterwards - do not calibrate with them.
#
#  IDENTIFIABILITY - read before adding parameters
#
#      q_evap = N_a * f * (pi/6)*D_d^3 * rho_v * h_fg,     N_a = (m*dT_sup)^n
#
#  `m` and `D_d^3` are MULTIPLICATIVELY DEGENERATE in this term: no amount of
#  boiling-curve data separates them. `D_d` does appear independently in `A_b`
#  (squared) and in the Cole frequency (as 1/sqrt(D_d)), but weakly. Fitting
#  `m`, `n`, `d_ref` and `K_ref` together yields a RIDGE, not a minimum - the
#  optimiser will report a confident answer that is one point on a valley floor.
#
#  So: `D_d` is fixed at the Fritz value for this fluid, and only `(m, n)` are
#  fitted. The grid search below is deliberately a grid rather than an optimiser
#  precisely so the ridge is visible rather than hidden.
# =============================================================================

using XCALibre, Printf
using XCALibre.ModelPhysics: BoilingState, solve_wall_temperature, single_phase_htc

# -----------------------------------------------------------------------------
# Operating point - must match the experimental curve
# -----------------------------------------------------------------------------
const P_SAT  = 0.4e6      # [Pa]
const U_BULK = 5.53       # [m/s]
const D_PIPE = 6.0e-3     # [m]
const Y_PLUS = 40.0       # first cell centre, from the mesh generator
const PR_T   = 0.85

const DATA_CSV = joinpath(@__DIR__, "data",
                          "tatsumoto2014_D6_L250_0.4MPa_5.53ms.csv")

# Only the NUCLEATE branch is a valid target: RPI has no CHF criterion and says
# nothing about film boiling. Fitting through the post-DNB points would drag the
# closure towards physics it cannot represent.
const Q_CHF = 64.0e3

# -----------------------------------------------------------------------------
# Fluid state
# -----------------------------------------------------------------------------
sat   = build_saturation_curve(H2(), p=(0.25e6, 1.25e6), T=(19.0, 33.0),
                               np=201, nT=201, verbose=false)
T_sat = saturation_temperature(sat, P_SAT)
h_fg  = latent_heat(sat, P_SAT, 0.0)
sigma = calculate_surface_tension(H2(), T_sat)
LIQ   = phase_properties_at(H2(), P_SAT, T_sat, branch=:liquid)
VAP   = phase_properties_at(H2(), P_SAT, T_sat, branch=:vapour)

# h_c from the same wall function the solver uses, with u_tau from Petukhov so
# this does not depend on a CFD run having converged.
const nu_l  = LIQ.mu/LIQ.rho
const Re    = U_BULK*D_PIPE/nu_l
const f_d   = (0.790*log(Re) - 1.64)^-2
const u_tau = U_BULK*sqrt(f_d/8)
const h_c_petukhov = single_phase_htc(Y_PLUS, u_tau, LIQ.rho, LIQ.cp, LIQ.mu, LIQ.k, PR_T)

# h_c FROM THE DATA, not from a correlation.
#
# The lowest-flux points of a boiling curve are single-phase convection: the wall
# is barely superheated, the site density is negligible, and q = h_c*dT_sup. So
# the data determines h_c directly, and it is the one quantity here that is NOT
# degenerate with (m, n) - it is fixed at the end of the branch where they do
# nothing.
#
# This matters. Petukhov gives 14199 W/m^2/K, which puts the lowest measured point
# at dT = q/h_c = 4320/14199 = 0.30 K against a MEASURED 0.663 K. Fitting (m, n)
# with an h_c that is 2x too high forces the site density to absorb the error, and
# the result is a fit that is wrong for a reason that has nothing to do with
# nucleation. The existing pipe notes record the same discrepancy from the other
# direction: a measured h_conv of 3692-5388 against a Dittus-Boelter ~10,200.
function h_c_from_data(dT, q; n_pts = 3)
    # Least squares through the origin on the lowest few points.
    k = min(n_pts, length(q))
    return sum(q[1:k].*dT[1:k])/sum(dT[1:k].^2)
end

# Fritz departure diameter for THIS fluid, rather than the 0.6 mm water value.
# theta is the contact angle in DEGREES - the 0.0208 coefficient expects degrees,
# a trap that has already caused one bug in this codebase.
# MEASURED contact angle for LH2 on this surface: 4 degrees.
#
# WAS 41.37, which is the KocamustafaogullariIshii library DEFAULT and a WATER
# value - K-I (1983) developed the correlation on water. Nothing justified it for
# hydrogen, whose surface tension is 0.9488 mN/m (~62x below water) and which
# wets metal almost perfectly. Same class of error as the shipped water site
# density fit (m = 210, n = 1.805) that was already replaced.
#
# D_d is LINEAR in theta, so this is a factor of 10.3:
#
#   theta      Fritz [um]    K-I [um]    Fritz/cell   K-I/cell
#   41.37        1110.17       12.474        19.93      0.2239
#    4.00         107.34        1.206         1.93      0.0217
#
# This INVERTS the earlier model choice. Fritz was rejected for giving 1.11 mm
# ("bubble 20x the cell") and K-I adopted to fix it - but that 1.11 mm came from
# the wrong angle. At 4 degrees Fritz gives 107 um, i.e. 1.93 cells, right in the
# 0.1-0.5 mm range measured for cryogen departure. K-I at 4 degrees gives 1.2 um,
# smaller than typical surface cavities, because its density-ratio correction
# 0.0012*(drho/rho_v)^0.9 is 0.0112 here against 0.92 at water conditions - 89x of
# work outside its calibration range.
const THETA_DEG = 4.0
const D_FRITZ = 0.0208*THETA_DEG*sqrt(sigma/(9.81*(LIQ.rho - VAP.rho)))

@info """Calibration setup
    p_sat, T_sat  : $(P_SAT/1e6) MPa, $(round(T_sat, digits=4)) K
    h_fg, sigma   : $(round(h_fg/1e3, digits=2)) kJ/kg, $(round(sigma*1e3, digits=4)) mN/m
    rho_l, rho_v  : $(round(LIQ.rho, digits=3)), $(round(VAP.rho, digits=3)) kg/m^3
    Re, u_tau     : $(round(Int, Re)), $(round(u_tau, digits=5)) m/s
    h_c Petukhov  : $(round(h_c_petukhov, digits=0)) W/m^2/K (superseded by the data-derived value below)
    D_d  Fritz    : $(round(D_FRITZ*1e6, digits=1)) um   (water default is 600 um)"""

# -----------------------------------------------------------------------------
# Experimental target
# -----------------------------------------------------------------------------
function load_curve(path)
    dT = Float64[]; q = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, '#') || startswith(s, "dT_sup")) && continue
        a, b = split(s, ',')
        push!(dT, parse(Float64, a)); push!(q, parse(Float64, b))
    end
    keep = q .<= Q_CHF
    p = sortperm(q[keep])
    return (dT[keep][p], q[keep][p])
end

const dT_EXP, q_EXP = load_curve(DATA_CSV)

# WHICH `h_c` THE FIT IS BUILT ON. This is a real fork, not a tuning knob.
#
#   :data           the experimental low-flux limit, ~7045 W/m^2/K. PHYSICALLY
#                   the better number - the data caps the convective coefficient
#                   at 4320/0.663 = 6516 and boiling only lowers that further.
#                   But the SOLVER does not use it, so a fit built on it leaves
#                   the CFD partition internally inconsistent.
#
#   :wall_function  `h_c_petukhov`, ~14199 W/m^2/K, i.e. exactly what
#                   `single_phase_htc` returns and therefore exactly what the RPI
#                   partition evaluates inside the CFD. MEASURED independently at
#                   13632 at y+ = 43 with the case's own properties (Pr = 1.63,
#                   cp = 18588, u_tau = 0.2325); Dittus-Boelter on the same
#                   properties gives 9076, and the 1.50x between them is the
#                   T+_bulk/T+_wall reference difference (30 vs 20), so the wall
#                   function is internally consistent - it is the LH2 data it
#                   disagrees with, by ~2x.
#
# CHOOSING `:wall_function` therefore makes the model SELF-CONSISTENT at the cost
# of folding a 2x convective over-prediction into the site density. The resulting
# (m, n) reproduces the curve in THIS solver, on THIS mesh, with THIS wall
# function, and will not transfer if any of those change. That is a deliberate
# trade, made because the CFD was running h_c ~ 13600 against a fit built on
# 7045 - the site density was compensating for a coefficient the solver never
# evaluates.
#
# The open physical question is why the wall function over-predicts against LH2
# by 2x. `Pr_t = 0.85` on a low-Pr cryogen and the very large near-saturation
# `cp` are the first places to look. Until that is settled, treat a
# `:wall_function` fit as compensation, not as nucleation physics.
# MEASURED, both ways. `:wall_function` FAILS - it is not a close second:
#
#   basis           h_c      fitted (m, n)          RMS(log dT)
#   :data           7045     3.0, 7.798 / 5.0, 11.018   0.0673
#   :wall_function  14199    1.113, 39.642              0.3085   <- 4.6x worse
#
# and it fails for a reason no refit can address: at q = 5.42 kW/m^2 the model
# returns dT = 0.382 K, which is exactly 5420/14199 - the PURE-CONVECTION
# asymptote - against a measured 0.784 K. Below the boiling threshold the site
# density does nothing, so dT = q/h_c is a hard floor and 14199 puts that floor
# above the data. The fit compensates by driving n to 39.6 to reach the
# high-flux end, which is both unusable (we fought runaway at n = 11.018) and
# meaningless. The experiment caps h_c at 4320/0.663 = 6516, and boiling only
# lowers it further.
#
# So the inconsistency is NOT resolvable by recalibration: the solver evaluates
# h_c ~ 13600 while this fit assumes 7045, and the two errors partially CANCEL -
# excess convection cools the wall, an inflated site density reheats it. That is
# why the CFD curve looks good and why 3e4 lands at 1.66 K against 1.58 K
# measured. Fixing it properly means fixing `single_phase_htc` for this fluid,
# not moving (m, n).
const H_C_BASIS = :data              # :data | :wall_function (see above - it fails)
const h_c = H_C_BASIS === :wall_function ?
    h_c_petukhov : h_c_from_data(dT_EXP, q_EXP)
@info "h_c basis" H_C_BASIS h_c=round(h_c, digits=0)
@info "convective coefficient" from_data=round(h_c, digits=0) petukhov=round(h_c_petukhov, digits=0) ratio=round(h_c/h_c_petukhov, digits=3)
@info "nucleate-branch target" n_points=length(q_EXP) q_range=(minimum(q_EXP), maximum(q_EXP)) dT_range=(minimum(dT_EXP), maximum(dT_EXP))

# -----------------------------------------------------------------------------
# Forward model: one parameter set -> the whole predicted curve
# -----------------------------------------------------------------------------
"""
Wall superheat predicted by the RPI partition at each experimental heat flux.

`T_l = T_sat`: the experiment is SATURATED boiling, so the near-wall liquid is at
saturation and there is no subcooling to carry.
"""
# WHICH DEPARTURE-DIAMETER MODEL. This is not a free choice once the mesh is
# fixed, because RPI deposits the vapour from a departing bubble into the FIRST
# CELL. Measured on the O-grid, first cell height 0.0557 mm:
#
#   TolubinskyKostanchuk(d_ref = Fritz)   D_d = 1.110 mm    D_d/dy = 19.9
#   KocamustafaogullariIshii              D_d = 0.0125 mm   D_d/dy = 0.22
#
# At D_d/dy = 20 a bubble's entire volume lands in a cell one twentieth its size,
# the local void saturates, the cell dries out and the wall runs away - measured
# on the staircase as alpha_max = 1.0 and T_max = 47.7 K at only 10 kW/m^2.
#
# Fritz is a POOL boiling correlation: it balances buoyancy against surface
# tension. At 5.53 m/s the bubble is sheared off long before buoyancy detaches
# it, so Fritz is the wrong physics here as well as the wrong number. K-I is a
# force balance that carries the density ratio, and it also responds to pressure
# (12.5 -> 1.0 um over 0.4 -> 1.1 MPa) where Tolubinsky-Kostanchuk is constant -
# which matters because the paper sweeps pressure.
#
# `q_evap ~ D_d^3`, so switching models changes the evaporative term by ~7e5 and
# the site density MUST be refitted. That is what `DEPARTURE` selects here.
DEPARTURE_MODEL = :tk      # :ki (Kocamustafaogullari-Ishii) or :tk (Tolubinsky)

departure(d_ref) = DEPARTURE_MODEL === :ki ?
    KocamustafaogullariIshii(theta_deg = THETA_DEG) :
    TolubinskyKostanchuk(d_ref = d_ref, d_max = 4*d_ref)

function predict(m, n; d_ref = D_FRITZ)
    rpi = RPI(patches = (:w,),
              site_density = LemmertChawla(m = m, n = n),
              departure_diameter = departure(d_ref))
    out = similar(q_EXP)
    for (i, q) in enumerate(q_EXP)
        s = BoilingState(T_w = T_sat, T_l = T_sat, T_sat = T_sat,
                         rho_l = LIQ.rho, rho_v = VAP.rho, cp_l = LIQ.cp,
                         k_l = LIQ.k, mu_l = LIQ.mu, sigma = sigma,
                         h_fg = h_fg, g = 9.81)
        T_w, _ = solve_wall_temperature(rpi, s, q, h_c)
        out[i] = T_w - T_sat
    end
    return out
end

"""
RMS residual in LOG superheat.

Log, because the branch spans a decade in flux for a factor of 2.5 in superheat -
an absolute residual would be dominated by the few highest-flux points. Also
guards against a parameter set that predicts zero or negative superheat.
"""
function cost(m, n; d_ref = D_FRITZ)
    dT = predict(m, n; d_ref = d_ref)
    any(x -> !(x > 0) || !isfinite(x), dT) && return Inf
    return sqrt(sum((log.(dT) .- log.(dT_EXP)).^2)/length(dT))
end

# -----------------------------------------------------------------------------
# Grid search
# -----------------------------------------------------------------------------
# A GRID, not an optimiser. The point is to see the shape of the objective: if it
# has a valley floor rather than a basin, `m` and `n` are trading off against each
# other and a single "best fit" would be misleading.
# n extended to 40. The first run of this script put the optimum at n = 14.000 -
# exactly the top of the old range - which means the grid, not the data, was
# choosing the answer. An optimum on a boundary is not an optimum. The range must
# be wide enough that the best point is INTERIOR, and this repo's own film-boiling
# notes already cite n ~ 21 for LH2, which the old grid could not even reach.
#
# `m` extends down with it: `N_a = (m*dT_sup)^n` at dT_sup ~ 1.6 K needs
# `m*dT_sup ~ (1e7)^(1/n)`, so a larger `n` demands a smaller `m`.
const M_GRID = 10.0 .^ range(-2.0, 4.0, length = 130)
const N_GRID = range(1.0, 40.0, length = 110)

function sweep()
    costs = fill(Inf, length(M_GRID), length(N_GRID))
    best = (cost = Inf, m = NaN, n = NaN)
    for (i, m) in enumerate(M_GRID), (j, n) in enumerate(N_GRID)
        c = cost(m, n)
        costs[i, j] = c
        c < best.cost && (best = (cost = c, m = m, n = n))
    end
    return best, costs
end

const best, costs = sweep()

@printf("\nBEST FIT   m = %.3f   n = %.3f   RMS(log dT) = %.4f\n",
        best.m, best.n, best.cost)

# How flat is the optimum? Everything within 10% of the best cost.
ridge = [(M_GRID[i], N_GRID[j]) for i in eachindex(M_GRID), j in eachindex(N_GRID)
         if costs[i, j] <= 1.1*best.cost]
if length(ridge) > 1
    ms = [r[1] for r in ridge]; ns = [r[2] for r in ridge]
    @printf("within 10%% of best: %d of %d grid points\n", length(ridge), length(costs))
    @printf("   m spans %.2f - %.2f   n spans %.3f - %.3f\n",
            minimum(ms), maximum(ms), minimum(ns), maximum(ns))
    println("   A WIDE span here means m and n are trading off - the fit is a")
    println("   ridge, and the individual values should not be quoted as physical.")
end

# -----------------------------------------------------------------------------
# The fitted curve against the data
# -----------------------------------------------------------------------------
dT_fit = predict(best.m, best.n)
dT_wat = predict(210.0, 1.805; d_ref = 0.6e-3)     # shipped water fit, for scale

@printf("\n%-12s %-12s %-12s %-12s %-10s\n",
        "q [kW/m2]", "dT_exp [K]", "dT_fit [K]", "dT_water [K]", "fit err")
println("-"^62)
for i in eachindex(q_EXP)
    @printf("%-12.2f %-12.3f %-12.3f %-12.3f %+-10.3f\n",
            q_EXP[i]/1e3, dT_EXP[i], dT_fit[i], dT_wat[i], dT_fit[i] - dT_EXP[i])
end

@printf("\nRMS(log dT):  fitted %.4f   shipped water fit %.4f\n",
        best.cost, cost(210.0, 1.805; d_ref = 0.6e-3))

println("""

USE THE RESULT LIKE THIS

    site_density       = LemmertChawla(m = $(round(best.m, digits=3)), n = $(round(best.n, digits=3))),
    departure_diameter = $(DEPARTURE_MODEL === :ki ?
        "KocamustafaogullariIshii(theta_deg = $THETA_DEG)" :
        "TolubinskyKostanchuk(d_ref = $(round(D_FRITZ, sigdigits=4)), d_max = $(round(4*D_FRITZ, sigdigits=4)))"),

CAVEATS

  * `m` is an AMPLITUDE that has absorbed everything degenerate with it - D_d^3,
    the Cole frequency constant, K_ref. It is not a physical site density and
    should not be quoted as one.
  * Valid ONLY up to CHF ($(Q_CHF/1e3) kW/m^2). RPI has no departure criterion and
    will extrapolate past it without complaint.
  * Calibrated at ONE pressure and ONE velocity. The paper sweeps both; a fit at
    0.4 MPa / $(U_BULK) m/s carries no guarantee at 0.7 or 1.1 MPa. Re-run this
    against those curves before trusting it there.
  * `h_c` = $(round(h_c, digits=0)) W/m^2/K basis = $(H_C_BASIS),
    data-derived = $(round(h_c_from_data(dT_EXP, q_EXP), digits=0)), Petukhov ($(round(h_c_petukhov, digits=0))). If your CFD produces a materially
    different h_conv the split between convection and evaporation moves, and this
    fit moves with it - re-run with the simulated value.
""")
