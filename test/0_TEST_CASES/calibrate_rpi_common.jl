# =============================================================================
#  Shared engine for the RPI calibration studies
# =============================================================================
#
#  `calibrate_rpi_lh2.jl` is the original full-range study and stands alone,
#  deliberately untouched. Everything after it shares this file so the fluid
#  state, the wall heat transfer coefficient and the objective cannot drift
#  between studies - a difference in the ANSWER must come from the parameter
#  being varied, not from the setup.
#
#  Include it, then call `run_study`.
# =============================================================================

using XCALibre, Printf
using XCALibre.ModelPhysics: BoilingState, solve_wall_temperature, single_phase_htc

# -----------------------------------------------------------------------------
# Operating point - matches the digitised curve
# -----------------------------------------------------------------------------
const P_SAT  = 0.4e6      # [Pa]
const U_BULK = 5.53       # [m/s]
const D_PIPE = 6.0e-3     # [m]
const Y_PLUS = 40.0
const PR_T   = 0.85
const Q_CHF  = 64.0e3     # nucleate branch ends here; RPI has no DNB criterion

const DATA_CSV = joinpath(@__DIR__, "data",
                          "tatsumoto2014_D6_L250_0.4MPa_5.53ms.csv")

const SAT   = build_saturation_curve(H2(), p=(0.25e6, 1.25e6), T=(19.0, 33.0),
                                     np=201, nT=201, verbose=false)
const T_SAT = saturation_temperature(SAT, P_SAT)
const H_FG  = latent_heat(SAT, P_SAT, 0.0)
const SIGMA = calculate_surface_tension(H2(), T_SAT)
const LIQ   = phase_properties_at(H2(), P_SAT, T_SAT, branch=:liquid)
const VAP   = phase_properties_at(H2(), P_SAT, T_SAT, branch=:vapour)

const NU_L  = LIQ.mu/LIQ.rho
const RE     = U_BULK*D_PIPE/NU_L
const F_D    = (0.790*log(RE) - 1.64)^-2
const U_TAU  = U_BULK*sqrt(F_D/8)
const H_C    = single_phase_htc(Y_PLUS, U_TAU, LIQ.rho, LIQ.cp, LIQ.mu, LIQ.k, PR_T)

# Fritz departure diameter for THIS fluid. theta is in DEGREES - the 0.0208
# coefficient expects degrees, a trap that has already caused one bug here.
const THETA_DEG = 4.0   # LH2 measured; see calibrate_rpi_lh2.jl
const D_FRITZ = 0.0208*THETA_DEG*sqrt(SIGMA/(9.81*(LIQ.rho - VAP.rho)))

"""
    load_curve(; q_min, q_max) -> (dT_sup, q_w)

Digitised nucleate branch, restricted to a flux window.

Windowing is the point of the low/high studies: the measured curve is NOT a
power law - the local exponent runs from about 1.8 below 20 kW/m^2 to about 11.5
above it - so a single `(m, n)` cannot fit both ends. Splitting the range lets
each half be fitted by a form that can actually represent it, at the cost of two
parameter sets and a stated crossover.
"""
function load_curve(; q_min = 0.0, q_max = Q_CHF)
    dT = Float64[]; q = Float64[]
    for line in eachline(DATA_CSV)
        s = strip(line)
        (isempty(s) || startswith(s, '#') || startswith(s, "dT_sup")) && continue
        a, b = split(s, ',')
        push!(dT, parse(Float64, a)); push!(q, parse(Float64, b))
    end
    keep = (q .>= q_min) .& (q .<= q_max)
    p = sortperm(q[keep])
    return (dT[keep][p], q[keep][p])
end

"""
    predict(site_density, d_ref, q_list) -> dT_sup

Wall superheat from the RPI partition at each heat flux. `T_l = T_sat`: the
experiment is SATURATED boiling, so there is no subcooling to carry.
"""
function predict(site_density, d_ref, q_list)
    rpi = RPI(patches = (:w,), site_density = site_density,
              departure_diameter = TolubinskyKostanchuk(d_ref = d_ref,
                                                        d_max = 4*d_ref))
    out = similar(q_list)
    for (i, q) in enumerate(q_list)
        s = BoilingState(T_w = T_SAT, T_l = T_SAT, T_sat = T_SAT,
                         rho_l = LIQ.rho, rho_v = VAP.rho, cp_l = LIQ.cp,
                         k_l = LIQ.k, mu_l = LIQ.mu, sigma = SIGMA,
                         h_fg = H_FG, g = 9.81)
        T_w, _ = solve_wall_temperature(rpi, s, q, H_C)
        out[i] = T_w - T_SAT
    end
    return out
end

"""
RMS residual in LOG superheat - the branch spans a decade in flux for a factor
of ~2.5 in superheat, so an absolute residual would be dominated by the few
highest-flux points.
"""
function cost(site_density, d_ref, q_list, dT_target)
    dT = predict(site_density, d_ref, q_list)
    any(x -> !(x > 0) || !isfinite(x), dT) && return Inf
    return sqrt(sum((log.(dT) .- log.(dT_target)).^2)/length(dT))
end

"""
    run_study(name, make_model, grids; q_min, q_max, fit_d_ref)

Sweep a parameter grid, report the best fit AND the shape of the objective.

`make_model(p...)` builds a site-density model from the swept parameters;
`grids` is a tuple of ranges, one per parameter, with `d_ref` appended last when
`fit_d_ref` is true.

A GRID, not an optimiser - deliberately. `m` and `D_d^3` are multiplicatively
degenerate in `q_evap`, so the objective has a valley rather than a basin. An
optimiser returns one point on the valley floor with no indication that the
neighbours are equally good; a grid shows the ridge, and the ridge is the honest
answer. The report flags when the optimum lands on a grid EDGE, which means the
fit has not converged and the quoted numbers are artefacts of the bounds.
"""
function run_study(name, make_model, grids; q_min = 0.0, q_max = Q_CHF,
                   fit_d_ref = false, d_ref_fixed = D_FRITZ)

    dT_t, q_t = load_curve(q_min = q_min, q_max = q_max)
    isempty(q_t) && error("no data points in $(q_min/1e3) - $(q_max/1e3) kW/m^2")

    println("\n", "="^70)
    println("  ", name)
    println("="^70)
    @printf("flux window : %.2f - %.2f kW/m^2   (%d points)\n",
            minimum(q_t)/1e3, maximum(q_t)/1e3, length(q_t))
    @printf("dT window   : %.3f - %.3f K\n", minimum(dT_t), maximum(dT_t))
    @printf("h_c         : %.0f W/m^2/K     D_d Fritz : %.1f um\n",
            H_C, D_FRITZ*1e6)

    idx = CartesianIndices(Tuple(length.(grids)))
    best = (cost = Inf, p = ntuple(_ -> NaN, length(grids)), I = first(idx))
    costs = fill(Inf, size(idx))
    for I in idx
        p = ntuple(k -> grids[k][I[k]], length(grids))
        d_ref = fit_d_ref ? p[end] : d_ref_fixed
        model_args = fit_d_ref ? p[1:end-1] : p
        c = cost(make_model(model_args...), d_ref, q_t, dT_t)
        costs[I] = c
        c < best.cost && (best = (cost = c, p = p, I = I))
    end

    @printf("\nBEST  RMS(log dT) = %.4f\n", best.cost)
    for k in eachindex(grids)
        edge = best.I[k] == 1 || best.I[k] == length(grids[k])
        @printf("   p%d = %-12.5g %s\n", k, best.p[k],
                edge ? "  <-- ON GRID EDGE: not converged, widen the range" : "")
    end

    within = count(<=(1.1*best.cost), costs)
    @printf("\nwithin 10%% of best : %d of %d grid points\n", within, length(costs))
    if within > 0.05*length(costs)
        println("   A large fraction means the parameters are TRADING OFF and the")
        println("   individual values are not physically meaningful on their own.")
    end

    dT_fit = predict(make_model((fit_d_ref ? best.p[1:end-1] : best.p)...),
                     fit_d_ref ? best.p[end] : d_ref_fixed, q_t)
    @printf("\n%-12s %-12s %-12s %-10s\n", "q [kW/m2]", "dT_exp", "dT_fit", "err")
    println("-"^50)
    for i in eachindex(q_t)
        @printf("%-12.2f %-12.3f %-12.3f %+-10.3f\n",
                q_t[i]/1e3, dT_t[i], dT_fit[i], dT_fit[i] - dT_t[i])
    end

    # Local exponent of the DATA, which is what a single power law has to match.
    if length(q_t) > 1
        nl = log(q_t[end]/q_t[1])/log(dT_t[end]/dT_t[1])
        @printf("\nlocal exponent of the data over this window : %.2f\n", nl)
    end

    return best
end
