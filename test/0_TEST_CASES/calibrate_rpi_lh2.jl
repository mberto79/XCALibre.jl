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
const h_c   = single_phase_htc(Y_PLUS, u_tau, LIQ.rho, LIQ.cp, LIQ.mu, LIQ.k, PR_T)

# Fritz departure diameter for THIS fluid, rather than the 0.6 mm water value.
# theta is the contact angle in DEGREES - the 0.0208 coefficient expects degrees,
# a trap that has already caused one bug in this codebase.
const THETA_DEG = 41.37
const D_FRITZ = 0.0208*THETA_DEG*sqrt(sigma/(9.81*(LIQ.rho - VAP.rho)))

@info """Calibration setup
    p_sat, T_sat  : $(P_SAT/1e6) MPa, $(round(T_sat, digits=4)) K
    h_fg, sigma   : $(round(h_fg/1e3, digits=2)) kJ/kg, $(round(sigma*1e3, digits=4)) mN/m
    rho_l, rho_v  : $(round(LIQ.rho, digits=3)), $(round(VAP.rho, digits=3)) kg/m^3
    Re, u_tau     : $(round(Int, Re)), $(round(u_tau, digits=5)) m/s
    h_c (y+=$(Y_PLUS))  : $(round(h_c, digits=0)) W/m^2/K
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
@info "nucleate-branch target" n_points=length(q_EXP) q_range=(minimum(q_EXP), maximum(q_EXP)) dT_range=(minimum(dT_EXP), maximum(dT_EXP))

# -----------------------------------------------------------------------------
# Forward model: one parameter set -> the whole predicted curve
# -----------------------------------------------------------------------------
"""
Wall superheat predicted by the RPI partition at each experimental heat flux.

`T_l = T_sat`: the experiment is SATURATED boiling, so the near-wall liquid is at
saturation and there is no subcooling to carry.
"""
function predict(m, n; d_ref = D_FRITZ)
    rpi = RPI(patches = (:w,),
              site_density = LemmertChawla(m = m, n = n),
              departure_diameter = TolubinskyKostanchuk(d_ref = d_ref,
                                                        d_max = 4*d_ref))
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
const M_GRID = 10.0 .^ range(-1.0, 3.0, length = 70)    # 0.1 -> 1000
const N_GRID = range(1.0, 14.0, length = 70)

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
    departure_diameter = TolubinskyKostanchuk(d_ref = $(round(D_FRITZ, sigdigits=4)),
                                              d_max = $(round(4*D_FRITZ, sigdigits=4))),

CAVEATS

  * `m` is an AMPLITUDE that has absorbed everything degenerate with it - D_d^3,
    the Cole frequency constant, K_ref. It is not a physical site density and
    should not be quoted as one.
  * Valid ONLY up to CHF ($(Q_CHF/1e3) kW/m^2). RPI has no departure criterion and
    will extrapolate past it without complaint.
  * Calibrated at ONE pressure and ONE velocity. The paper sweeps both; a fit at
    0.4 MPa / $(U_BULK) m/s carries no guarantee at 0.7 or 1.1 MPa. Re-run this
    against those curves before trusting it there.
  * `h_c` here comes from Petukhov, not from your CFD. If the simulated `h_conv`
    differs materially from $(round(h_c, digits=0)) W/m^2/K, re-run with that value:
    the split between convection and evaporation depends on it.
""")
