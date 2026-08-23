# =============================================================================
#  LH2 boiling curve: a heat-flux STAIRCASE on the Tatsumoto pipe
# =============================================================================
#
#  Runs the same geometry, fluid and models as
#  `3d_LH2_pipe_forced_convection.jl`, but sweeps the wall heat flux so one run
#  produces a whole boiling curve instead of a single operating point.
#
#      2 flow-throughs   settle at Q_SCHEDULE[1]        (discarded)
#      1 flow-through    at each subsequent Q            (recorded)
#
#  The comparison against Tatsumoto et al. (2014) is `T_wall - T_sat` versus
#  `q_w`, i.e. their Figs. 3-4.
#
#  WHY A STAIRCASE AND NOT SEPARATE RUNS
#
#  Each plateau starts from the converged state of the previous one, so only the
#  first needs a full development transient. Separate runs would each pay the two
#  flow-throughs of settling. It also mirrors how the experiment was actually
#  done - the heater power is stepped and the wall temperature allowed to settle.
#
#  ONE FLOW-THROUGH MAY NOT BE ENOUGH. It is sufficient for the near-wall thermal
#  and vapour fields, which respond on the near-wall residence time (much shorter
#  than the bulk one), but check the reported `T_wall` at successive write
#  intervals within a plateau: if it is still drifting at the end, raise
#  `FLOW_THROUGHS_PER_STEP`. That check is the point of the per-interval logging.
#
#  COST. At dt = 2e-6 with L_total/U = 58 ms, one flow-through is ~29,000 steps.
#  Two for settling plus one per level: with 8 levels that is ~290,000 steps.
#  Plan for an overnight run and use `write_interval` to keep an eye on it.
# =============================================================================

using XCALibre
using Printf

# -----------------------------------------------------------------------------
# DRIFT TREATMENT - diagnostic toggle
# -----------------------------------------------------------------------------
# The drift term div[alpha*(1-alpha)*Urdotf] is EXPLICIT by default: built as a
# flux and its divergence added to S_alpha. Cheap, but it carries its own
# stability limit and was recorded as diverging at 2e4 under vapour tracking.
#
# WHY TEST IT NOW. The drift diameter was raised 12.5 -> 429 um to fix vapour
# piling up at the wall, which multiplied Ur by 373x - so an explicitly treated
# source with a known stability limit just got 373x stronger. The oscillation
# shows up at alpha ~ 0.5-0.7, which is exactly where alpha*(1-alpha) is maximal
# and the explicit source is therefore largest.
#
# Setting this routes the term through the Picard linearisation onto the matrix
# diagonal instead. If the oscillation clears, the cause is explicit-treatment
# stability, not the drift physics.
#
# COST: the solver notes record ~3x the per-step cost, and "a 1500-step run had
# not finished in 80 minutes" at 2e4. Budget accordingly, and consider dropping
# Q_SCHEDULE to the single level being diagnosed.
#
# `ENV` is read at runtime inside `advance_alpha_implicit!`, so setting it here -
# before the include and before any `run!` - is sufficient.
# RESULT of that diagnostic: the drift DISCRETISATION is not the cause. Levels
# 5e3/1e4/2e4 came out BIT-IDENTICAL to the explicit run (both converge to the
# same steady state, as two consistent discretisations of one term must), and at
# 3e4 the implicit form was slightly WORSE - alpha_max 0.6915 -> 1.0. So it is
# back off: it costs ~3x per step and buys nothing.
const DRIFT_IMPLICIT = false
if DRIFT_IMPLICIT
    ENV["DRIFT_IMPLICIT"] = "1"
    @info "DRIFT: implicit (Picard, on the diagonal) - diagnostic run"
else
    delete!(ENV, "DRIFT_IMPLICIT")
    @info "DRIFT: explicit (default)"
end

# -----------------------------------------------------------------------------
# Heat flux schedule
# -----------------------------------------------------------------------------
# The paper's developed nucleate boiling regime for D6_L250 spans roughly
# 1e4 - 1e5 W/m^2, with DNB near 6e4 at 5.33 m/s (Fig. 5b). The schedule stays
# BELOW DNB: RPI models nucleate boiling only and says nothing about the film
# boiling branch, so points above DNB would be extrapolation, not validation.
#
# Geometric spacing gives even coverage on the log-log axes a boiling curve is
# normally plotted on.
# const Q_SCHEDULE = [4e3, 6e3, 8e3, 1.0e4, 1.5e4, 2.2e4]   # [W/m^2]
# const Q_SCHEDULE = [3.5e4, 5.0e4, 6.4e4, 7.0e4, 8.0e4, 1.0e5]   # [W/m^2]
# FILM-BOILING LADDER. Extends past the 3e4 ceiling that RPI-only reached: 4e4
# was the first level to fail without a departure criterion, so the levels above
# it are exactly what `FILM_BOILING = true` is being added to reach. 6.4e4 is the
# MEASURED CHF for this case, so 5e4/6.4e4 straddle departure and 7e4 sits past
# it on the film branch.
#
# Re-verifying 1e4-3e4 is deliberate, not padding: the blend must leave the
# already-validated nucleate branch UNCHANGED. If those three move, `alpha_1` is
# too low and is eating into nucleate boiling - that check is the whole point of
# running them again.
# const Q_SCHEDULE = [5.0e3, 1.0e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4, 6.4e4, 7.0e4]   # [W/m^2]
const Q_SCHEDULE = [5.0e3, 1.0e4, 2.0e4, 3.0e4]   # RPI-only ceiling (pre-film)

const FLOW_THROUGHS_INIT = 1      # settling at Q_SCHEDULE[1], discarded
const FLOW_THROUGHS_PER_STEP = 1  # at each level, including the first

# -----------------------------------------------------------------------------
# Everything up to (but not including) the time loop comes from the single-point
# case, so the two cannot drift apart. It defines: mesh_dev, model, BCs, schemes,
# solvers, T_sat, U_inlet_mag, L_total, D, WALL_HEAT_FLUX, hardware.
#
# `BOILING_CURVE_SETUP_ONLY` tells that file to stop before `run!`.
# -----------------------------------------------------------------------------
const BOILING_CURVE_SETUP_ONLY = true
include(joinpath(@__DIR__, "3d_LH2_pipe_forced_convection.jl"))

# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------
const DT = 2.0e-5
const T_FLOW_THROUGH = L_total/U_inlet_mag
const STEPS_PER_FT = round(Int, T_FLOW_THROUGH/DT)

@info """Boiling curve schedule
    flow-through time : $(round(T_FLOW_THROUGH*1e3, digits=2)) ms  ($(STEPS_PER_FT) steps at dt = $DT s)
    initialisation    : $(FLOW_THROUGHS_INIT) FT at $(Q_SCHEDULE[1]/1e3) kW/m^2
    levels            : $(length(Q_SCHEDULE))  ($(join(round.(Q_SCHEDULE./1e3, digits=1), ", ")) kW/m^2)
    total steps       : $((FLOW_THROUGHS_INIT + FLOW_THROUGHS_PER_STEP*length(Q_SCHEDULE))*STEPS_PER_FT)"""

"""
Rebuild the boundary conditions with a new heated-wall flux.

Only `T` on `:pipeWall` changes; everything else is copied from the setup file so
the two cannot diverge. `assign` is cheap, and rebuilding is safer than mutating
a `FixedHeatFlux` in place - the value is captured by the solver at setup, so an
in-place edit may or may not be seen depending on when it happens.
"""
function boundaries_at(q_w)
    noSlip = [0.0, 0.0, 0.0]
    return assign(region = mesh_dev, (
        U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet),
             Wall(:pipeWall, noSlip), Wall(:wallUnheated, noSlip),
             Symmetry(:symmetryX), Symmetry(:symmetryY)],
        p_rgh = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0),
                 Zerogradient(:pipeWall), Zerogradient(:wallUnheated),
                 Symmetry(:symmetryX), Symmetry(:symmetryY)],
        alpha = [Dirichlet(:inlet, 0.0), Zerogradient(:outlet),
                 Zerogradient(:pipeWall), Zerogradient(:wallUnheated),
                 Symmetry(:symmetryX), Symmetry(:symmetryY)],
        T = [Dirichlet(:inlet, T_inlet), Zerogradient(:outlet),
             FixedHeatFlux(:pipeWall, q_w), Zerogradient(:wallUnheated),
             Symmetry(:symmetryX), Symmetry(:symmetryY)],
        k = [Dirichlet(:inlet, k_inlet), Zerogradient(:outlet),
             KWallFunction(:pipeWall), KWallFunction(:wallUnheated),
             Symmetry(:symmetryX), Symmetry(:symmetryY)],
        omega = [Dirichlet(:inlet, omega_inlet), Zerogradient(:outlet),
                 OmegaWallFunction(:pipeWall), OmegaWallFunction(:wallUnheated),
                 Symmetry(:symmetryX), Symmetry(:symmetryY)],
        nut = [Dirichlet(:inlet, nut_inlet), Zerogradient(:outlet),
               NutWallFunction(:pipeWall), NutWallFunction(:wallUnheated),
               Symmetry(:symmetryX), Symmetry(:symmetryY)],
    ))
end

# ADAPTIVE TIME STEPPING, keyed on the ALPHA Courant number.
#
# Measured 2026-08-19 on the single-point case, cold start, 60 steps at DT = 2e-5,
# sweeping only the wall flux:
#
#   q_w = 10 kW/m^2   Courant 0.29     AlphaCourant 0      p_rgh 3.5e-11  healthy
#   q_w = 20 kW/m^2   Courant 11.8     AlphaCourant 4.19   p_rgh 1.0e-9   marginal
#
# Note WHICH number goes bad. At 20 kW/m^2 the pressure residual is still 1e-9 -
# the pressure solve is fine - while the ALPHA Courant number is 4.19, i.e. over
# four times its stability limit. `alpha_transport = :mules` is an EXPLICIT
# flux-corrected update, so it carries a hard Courant condition that nothing else
# in the solver does, and vapour only appears in quantity once the flux is high
# enough to boil. That is why the sweep survives initialisation and the first
# level (10 kW/m^2, essentially no vapour) and then loses the second.
#
# A fixed DT cannot serve both: small enough for the boiling levels wastes most of
# its time on the single-phase ones. `maxAlphaCo = 0.25` lets the step follow the
# vapour, and `maxCo` is set loose because the pressure path is demonstrably not
# the constraint here.
const ADAPTIVE = AdaptiveTimeStepping(
    maxCo      = 0.5,     # not the binding constraint - see above
    maxAlphaCo = 0.25,    # MULES stability; this is the one that bites
    minShrink  = 0.1,
    maxGrow    = 1.1)     # rise slowly: a level that has just settled should not
                          # be kicked by a sudden step increase

function config_at(q_w, n_steps; write_interval, adaptive = ADAPTIVE)
    return Configuration(
        solvers = solvers, schemes = schemes,
        runtime = Runtime(iterations = n_steps, time_step = DT,
                          write_interval = write_interval,
                          adaptive = adaptive),
        hardware = hardware, boundaries = boundaries_at(q_w))
end

# -----------------------------------------------------------------------------
# Initial fields
# -----------------------------------------------------------------------------
initialise!(model.momentum.U, velocity)
initialise!(model.fluid.p_rgh, 0.0)
initialise!(model.fluid.alpha, 0.0)
initialise!(model.energy.T, T_inlet)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, omega_inlet)
initialise!(model.turbulence.nut, nut_inlet)

# =============================================================================
# Stage 1: initialisation
# =============================================================================
@info "=== INITIALISATION: $(FLOW_THROUGHS_INIT) flow-throughs at $(Q_SCHEDULE[1]/1e3) kW/m^2 ==="
run!(model, config_at(Q_SCHEDULE[1], FLOW_THROUGHS_INIT*STEPS_PER_FT;
                      write_interval = FLOW_THROUGHS_INIT*STEPS_PER_FT), inner_loops = 5)

# =============================================================================
# Stage 2: the staircase
# =============================================================================
# `run!` continues from the current state of `model`, so each level starts where
# the previous one finished.
#
# The wall state is reported by the solver itself (`report_wall_boiling`) at
# every write interval, so set the interval to a fraction of a plateau and read
# `T_wall` from the log: if it is still moving at the end of a level, that level
# has not settled and `FLOW_THROUGHS_PER_STEP` needs raising.
const WRITE_EVERY = max(1, STEPS_PER_FT ÷ 4)

# Results are COLLECTED here rather than read out of the log. The solver stashes
# each wall report in `LAST_WALL_REPORT` at every write interval, so the value
# available immediately after `run!` is the end-of-level state.
const RESULTS = NamedTuple[]

for (i, q_w) in enumerate(Q_SCHEDULE)
    @info "=== LEVEL $i/$(length(Q_SCHEDULE)): q_w = $(q_w/1e3) kW/m^2 ==="
    run!(model, config_at(q_w, FLOW_THROUGHS_PER_STEP*STEPS_PER_FT;
                          write_interval = WRITE_EVERY), inner_loops = 5)

    T = model.energy.T.values
    a = model.fluid.alpha.values
    r = LAST_WALL_REPORT[]

    if r === nothing
        @warn "level $i: no wall report - is `wall_boiling` active?" q_w
    else
        push!(RESULTS, (q_w = q_w, dT_sup = r.dT_sup, T_wall = r.T_wall,
                        q_conv = r.q_conv, q_quench = r.q_quench, q_evap = r.q_evap,
                        closure = r.closure, evap_frac = r.evap_frac,
                        alpha_max = maximum(a), alpha_min = minimum(a)))
    end

    @info(
        "level $i complete",
        q_w = q_w,
        T_max = maximum(T), T_min = minimum(T),
        dT_max_vs_Tsat = maximum(T) - T_sat,
        # ALPHA_MAX IS THE ONE TO WATCH. `liquid_phase = 2` in the setup file, so
        # `alpha` is the VOID fraction: `alpha = 0` is pure liquid, `alpha = 1` is
        # pure vapour. `alpha_min` is therefore ~0 in any run that has liquid
        # anywhere, which is every run, and says nothing about whether the wall is
        # boiling. `alpha_max` is the peak void - i.e. whether vapour is being
        # generated at all, and how much.
        alpha_max = maximum(a),
        alpha_min = minimum(a),
        finite = all(isfinite, T) && all(isfinite, a),
    )
end

# -----------------------------------------------------------------------------
# Write the simulated curve next to the experimental one
# -----------------------------------------------------------------------------
const RESULTS_CSV = joinpath(@__DIR__, "data", "boiling_curve_simulated.csv")

open(RESULTS_CSV, "w") do io
    println(io, "# XCALibre RPI boiling curve")
    println(io, "# CASE=$CASE  p_sat=$(p_sat/1e6) MPa  U=$U_inlet_mag m/s  T_sat=$(round(T_sat, digits=4)) K")
    println(io, "# dt=$DT  flow-throughs per level=$FLOW_THROUGHS_PER_STEP")
    println(io, "q_w,dT_sup,T_wall,q_conv,q_quench,q_evap,closure,evap_frac,alpha_max,alpha_min")
    for r in RESULTS
        # TEN values for TEN headers, and `alpha_max` BEFORE `alpha_min` to match.
        # This previously wrote nine values ending in `r.alpha_min`, so the column
        # labelled `alpha_max` actually held `alpha_min` and `alpha_max` was never
        # written at all - i.e. the CSV reported the one void diagnostic the note
        # above says "says nothing about whether the wall is boiling", and dropped
        # the one it calls THE ONE TO WATCH. Symptom: an `alpha_max` column full of
        # ~1e-64 while the real peak void at q_w = 1e4 is 6.1e-2.
        @printf(io, "%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                r.q_w, r.dT_sup, r.T_wall, r.q_conv, r.q_quench, r.q_evap,
                r.closure, r.evap_frac, r.alpha_max, r.alpha_min)
    end
end
@info "simulated curve written" RESULTS_CSV n_levels=length(RESULTS)

# =============================================================================
# Comparison against the experiment
# =============================================================================
#
#  Digitised curve: p_sat = 0.4 MPa, U = 5.53 m/s, D6_L250 - matching `p_sat`,
#  `U_inlet_mag` and `CASE` in the setup file. Change any of those and this
#  comparison stops being valid.
#
#  READ FROM THE DATA (not from the paper's text):
#    nucleate boiling  q = 4.3 -> 64 kW/m2 with dT_sup only 0.66 -> 1.67 K
#    DNB / CHF         ~64-69 kW/m2, where dT_sup departs
#    film boiling      dT_sup -> 100 K while q reaches only 147 kW/m2
#
#  `Q_SCHEDULE` therefore stops at 50 kW/m2, safely below CHF. RPI models
#  nucleate boiling only; past DNB it is extrapolation, not validation.
#
#  THE HARD PART. The nucleate branch is nearly VERTICAL - a 15x change in heat
#  flux moves the wall superheat by about one kelvin. Reproducing that demands
#  the superheat to within a few tenths of a K, which is a far sharper test than
#  a boiling curve usually looks. Expect the shipped closures to miss it:
#  `LemmertChawla` (m = 210, n = 1.805) and `TolubinskyKostanchuk` (0.6 mm, 45 K)
#  are WATER fits with no established cryogenic values. Disagreement here is the
#  starting point for calibration, not evidence of a coding error.

const EXPERIMENT_CSV = joinpath(@__DIR__, "data",
                                "tatsumoto2014_D6_L250_0.4MPa_5.53ms.csv")

"""
    load_experiment(path) -> (dT_sup, q_w)

Read the digitised boiling curve, skipping `#` comments and the header row.
"""
function load_experiment(path)
    dT = Float64[]; q = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, '#') || startswith(s, "dT_sup")) && continue
        a, b = split(s, ',')
        push!(dT, parse(Float64, a)); push!(q, parse(Float64, b))
    end
    p = sortperm(q)
    return (dT[p], q[p])
end

"""
    compare_to_experiment(sim; path=EXPERIMENT_CSV)

Compare simulated `(q_w, dT_sup)` pairs against the digitised curve, reporting
the experimental superheat interpolated to each simulated heat flux.

`sim` is a vector of `(q_w, dT_sup)` tuples. Take `dT_sup` from the
`report_wall_boiling` line the solver logs at the END of each level - that is
the area-averaged wall superheat, which is what the experiment measured.

Only the NUCLEATE branch is compared; simulated points above CHF are flagged and
skipped, since RPI has nothing to say there.
"""
function compare_to_experiment(sim; path = EXPERIMENT_CSV, q_chf = 64.0e3)
    dT_e, q_e = load_experiment(path)
    @printf("\n%-12s %-14s %-14s %-10s\n", "q_w [kW/m2]", "dT_sim [K]", "dT_exp [K]", "diff [K]")
    println("-"^54)
    for (q, dT_s) in sim
        if q > q_chf
            @printf("%-12.1f %-14.3f %-14s %-10s\n", q/1e3, dT_s, "-", "above CHF")
            continue
        end
        i = searchsortedfirst(q_e, q)
        i = clamp(i, 2, length(q_e))
        w = (q - q_e[i-1])/(q_e[i] - q_e[i-1])
        dT_x = dT_e[i-1] + w*(dT_e[i] - dT_e[i-1])
        @printf("%-12.1f %-14.3f %-14.3f %+-10.3f\n", q/1e3, dT_s, dT_x, dT_s - dT_x)
    end
    return nothing
end

# Run the comparison automatically on what the staircase collected.
if isempty(RESULTS)
    @warn "No levels recorded - nothing to compare."
else
    compare_to_experiment([(r.q_w, r.dT_sup) for r in RESULTS])
end

# -----------------------------------------------------------------------------
# Boiling curve figure
# -----------------------------------------------------------------------------
# `Plots` is imported HERE, in the case script, not by XCALibre - a solver
# library should not force a plotting stack on everyone who loads it.
#
# Conventional boiling-curve axes: wall superheat on x, heat flux on y, both
# logarithmic. The experiment's nucleate branch is nearly VERTICAL on these axes
# (15x in q for ~1 K in superheat), so a linear y axis would compress the whole
# comparison into the top of the frame.
using Plots

if isempty(RESULTS)
    @warn "No levels recorded - no figure produced."
else
    dT_exp, q_exp = load_experiment(EXPERIMENT_CSV)
    dT_sim = [r.dT_sup for r in RESULTS]
    q_sim  = [r.q_w    for r in RESULTS]

    # Split the experiment at CHF: past it the physics is film boiling, which RPI
    # does not model. Showing it in a different style keeps the comparison honest
    # about which points are a fair test.
    const Q_CHF = 64.0e3
    nucleate = q_exp .<= Q_CHF

    plt = plot(dT_exp[nucleate], q_exp[nucleate];
        seriestype = :scatter, marker = (:circle, 5), color = :black,
        label = "Tatsumoto 2014 (nucleate)",
        xscale = :log10, yscale = :log10,
        xlabel = "wall superheat  T_w - T_sat  [K]",
        ylabel = "wall heat flux  q_w  [W/m²]",
        title = "LH2 forced convection boiling, $(p_sat/1e6) MPa, $(U_inlet_mag) m/s",
        legend = :bottomright, framestyle = :box, minorgrid = true,
        size = (760, 620), dpi = 200)

    plot!(plt, dT_exp[.!nucleate], q_exp[.!nucleate];
        seriestype = :scatter, marker = (:xcross, 5), color = :grey,
        label = "Tatsumoto 2014 (post-DNB, not modelled)")

    hline!(plt, [Q_CHF]; linestyle = :dash, color = :grey, label = "CHF ≈ 64 kW/m²")

    plot!(plt, dT_sim, q_sim;
        marker = (:diamond, 6), color = :crimson, linewidth = 2,
        label = "XCALibre RPI")

    figpath = joinpath(@__DIR__, "data", "boiling_curve.png")
    savefig(plt, figpath)
    @info "boiling curve figure written" figpath
    display(plt)
end
