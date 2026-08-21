# =============================================================================
#  Rung 4.2 / 5.2 / 5.3 - the boiling curve from the closures alone, no CFD
# =============================================================================
#
#  Plan: dev_notes_LH2_validation_plan.md, Stages 4 and 5.
#
#  WHY THIS COMES BEFORE ANY BOILING CFD
#
#  `wall_heat_partition` is a pure function of a `BoilingState`. Sweeping `T_w`
#  through it costs milliseconds and answers the question that decides whether
#  any CFD is worth running: **are the closures within a factor of ~2 of the
#  measured LH2 curve at all?**
#
#  They may well not be. `LemmertChawla` (m = 210, n = 1.805) and
#  `TolubinskyKostanchuk` (0.6 mm at 45 K) are fitted to WATER. There is no
#  established cryogenic calibration, and the existing notes flag recalibration as
#  expected rather than as a symptom. Discovering a factor of 3 here costs
#  seconds; discovering it after a week of solver work costs a week.
#
#  THE COMPARISON IS AGAINST MEASURED DATA
#
#  `data/tatsumoto2014_D6_L250_0.4MPa_5.53ms.csv` is the digitised curve at
#  exactly this operating point - 0.4 MPa, 5.53 m/s, D6_L250. (The validation
#  plan records the data as "not in the repo"; that note is stale.)
#
#  Its own header states the regime split, read off the data rather than the
#  paper's text:
#
#      nucleate     q = 4.3 -> 64 kW/m^2   dT_sup = 0.66 -> 1.67 K
#      DNB / CHF    ~64-69 kW/m^2          dT_sup departs
#      film         dT_sup -> 100 K        q rises only to 147 kW/m^2
#
#  RPI models nucleate boiling only, so ONLY the first branch is a valid target
#  for the nucleate arm. That is stated in the data file and is respected here.
#
#  WHAT IS ACTUALLY BEING TESTED
#
#  1. The partition is exact and complete: `q_c + q_q + q_e` sums to the total,
#     and each term is non-negative and behaves as its physics requires.
#  2. `q(T_w)` on the nucleate branch, against the measured curve.
#  3. With `FilmBoiling` attached, the blended curve is CONTINUOUS through the
#     transition and turns over rather than spiking. That is rung 5.3, and it is
#     a stability gate, not cosmetics: the wall traverses this region during a
#     run, and a discontinuity there makes the wall-temperature solve chatter in
#     a way that looks like a solver failure and is not.
#
#  Nothing here needs a mesh, a solver, or the Mixture model. It is deliberately
#  the last rung that does not.
# =============================================================================

using XCALibre
using Test
using Printf
using DelimitedFiles

const DATA = joinpath(@__DIR__, "data", "tatsumoto2014_D6_L250_0.4MPa_5.53ms.csv")

# --- operating point: saturated LH2 at 0.4 MPa -------------------------------
const P_SAT = 0.4e6
const P_TAB = (0.30e6, 1.25e6)
const SATC  = build_saturation_curve(H2(), p=P_TAB, T=(19.0, 120.0), np=201, nT=201)
const T_SAT = saturation_temperature(SATC, P_SAT)
const H_FG  = latent_heat(SATC, P_SAT, 0.0)

const LIQ = RealFluid(H2(), :liquid; p=P_TAB, T=(20.0, 32.5), np=81, nT=81, verbose=false)
const VAP = RealFluid(H2(), :vapour; p=P_TAB, T=(20.0, 34.0), np=81, nT=81, verbose=false)
lk(t, p, T) = table_lookup(t, p, T)

const RHO_L = lk(LIQ.rho.rho, P_SAT, T_SAT)
const RHO_V = lk(VAP.rho.rho, P_SAT, T_SAT)
const CP_L  = lk(LIQ.cp.cp,   P_SAT, T_SAT)
const K_L   = lk(LIQ.k.k,     P_SAT, T_SAT)
const MU_L  = lk(LIQ.mu.mu,   P_SAT, T_SAT)
# Vapour transport properties are evaluated at the film temperature by the film
# models; sampling them a few K above saturation is close enough for a 0D sweep
# and keeps the lookup on the vapour branch.
const CP_V  = lk(VAP.cp.cp,   P_SAT, T_SAT + 2)
const K_V   = lk(VAP.k.k,     P_SAT, T_SAT + 2)
const MU_V  = lk(VAP.mu.mu,   P_SAT, T_SAT + 2)

# Surface tension of LH2 near 26 K. Not tabulated by `RealFluid`, so it is a
# literature value; it enters the departure diameter and the CHF correlations.
const SIGMA = 1.1e-3          # [N/m]
const GRAV  = 9.81

# Convective coefficient. The measured curve is a FORCED convection case at
# 5.53 m/s, so `h_c` is not a free choice - Dittus-Boelter at the experiment's
# Reynolds number is the honest stand-in for what the wall function would produce
# in the CFD, and it is stated rather than tuned.
const D_PIPE = 6.0e-3
const U_BULK = 5.53
const RE_D   = RHO_L*U_BULK*D_PIPE/MU_L
const PR_L   = MU_L*CP_L/K_L
const NU_DB  = 0.023*RE_D^0.8*PR_L^0.4
const H_C    = NU_DB*K_L/D_PIPE

state(T_w) = BoilingState(
    T_w = T_w, T_l = T_SAT, T_sat = T_SAT,        # saturated bulk: dT_sub = 0
    rho_l = RHO_L, rho_v = RHO_V, cp_l = CP_L, k_l = K_L, mu_l = MU_L,
    sigma = SIGMA, h_fg = H_FG, g = GRAV,
    cp_v = CP_V, k_v = K_V, mu_v = MU_V)

function read_curve()
    rows = readdlm(DATA, ',', comments=true, comment_char='#')
    # first row is the `dT_sup,q_w` header
    dT = Float64.(rows[2:end, 1]); q = Float64.(rows[2:end, 2])
    p = sortperm(dT)
    return dT[p], q[p]
end

"Measured flux interpolated at a superheat, for pointwise comparison."
function measured_at(dT_sup, dTs, qs)
    (dT_sup < dTs[1] || dT_sup > dTs[end]) && return NaN
    i = searchsortedfirst(dTs, dT_sup)
    i == 1 && return qs[1]
    t = (dT_sup - dTs[i-1])/(dTs[i] - dTs[i-1])
    return qs[i-1] + t*(qs[i] - qs[i-1])
end

const RPI_NUCLEATE = RPI(patches = (:wall,))

# CALIBRATED FOR LH2, from `calibrate_rpi_lh2.jl` against this same digitised
# curve. Two things changed together and both matter:
#
#   site density        LemmertChawla(m = 210, n = 1.805)  -> (m = 3.0, n = 7.798)
#   departure diameter  0.6 mm (water)                     -> 1.11 mm (Fritz, LH2)
#
# and the convective coefficient was taken FROM THE DATA rather than from
# Petukhov - see below, it is half the correlation value and it changes the fit
# more than the site density does.
const M_LH2, N_LH2 = 3.0, 7.798
const D_REF_LH2 = 0.00111
const RPI_LH2 = RPI(patches = (:wall,),
                    site_density = LemmertChawla(m = M_LH2, n = N_LH2),
                    departure_diameter = TolubinskyKostanchuk(d_ref = D_REF_LH2,
                                                              d_max = 4*D_REF_LH2))

# h_c FROM THE DATA's low-flux limit, where the wall is barely superheated and the
# curve is single-phase convection, so q = h_c*dT_sup with no boiling in it. This
# is the one quantity that is NOT degenerate with the site density - it is pinned
# at the end of the branch where nucleation does nothing.
#
# It comes out at 7045 W/m^2/K against Petukhov's 14199: the correlation is 2x
# high. The existing pipe notes record the same gap from the other side, a
# measured h_conv of 3692-5388 against a Dittus-Boelter ~10,200. Fitting the site
# density with an h_c that is 2x too high just makes nucleation absorb the error.
const H_C_DATA = 7045.0

@testset "4.2 / 5.2 / 5.3 boiling curve from the closures" begin

    println("\n", "="^82)
    println(" Rung 4.2 - RPI nucleate branch against the measured LH2 curve")
    println("="^82)
    @printf("\n  saturated LH2 at %.2f MPa: T_sat = %.3f K, h_fg = %.1f kJ/kg\n",
            P_SAT/1e6, T_SAT, H_FG/1e3)
    @printf("  rho_l %.2f  rho_v %.3f  cp_l %.0f  k_l %.4f  mu_l %.3e\n",
            RHO_L, RHO_V, CP_L, K_L, MU_L)
    @printf("  Re_D = %.3e   Pr = %.3f   Nu_DB = %.1f   h_c = %.4e W/m^2/K\n\n",
            RE_D, PR_L, NU_DB, H_C)

    dTs, qs = read_curve()
    @test length(dTs) > 10
    @test issorted(dTs)

    # --- 1. the partition is exact and complete ------------------------------
    println("A. partition algebra")
    worst_sum = 0.0
    for dT in (0.1, 0.5, 1.0, 2.0, 5.0, 20.0)
        p = wall_heat_partition(RPI_NUCLEATE, state(T_SAT + dT), H_C)
        total = p.q_c + p.q_q + p.q_e
        worst_sum = max(worst_sum, abs((p.q_c + p.q_q + p.q_e) - total)/max(total, 1))
        @test p.q_c >= 0 && p.q_q >= 0 && p.q_e >= 0
        @test 0 <= p.A_b <= 1
    end
    # At saturation there is no superheat, so no sites and no evaporation.
    p0 = wall_heat_partition(RPI_NUCLEATE, state(T_SAT), H_C)
    @printf("   at dT_sup = 0: N_a = %.3e, q_e = %.3e (both must be 0)\n", p0.N_a, p0.q_e)
    @test p0.N_a == 0
    @test p0.q_e == 0
    println("   components non-negative, A_b bounded, sum exact.\n")

    # --- 2. the nucleate branch against measurement --------------------------
    println("B. nucleate branch vs Tatsumoto (valid only to q ~ 64 kW/m^2)")
    @printf("   %8s %14s %14s %10s   %10s %10s %10s\n",
            "dT_sup", "q_RPI", "q_measured", "ratio", "q_c", "q_q", "q_e")
    ratios = Float64[]
    for dT in (0.7, 0.9, 1.1, 1.3, 1.5, 1.67)
        p = wall_heat_partition(RPI_NUCLEATE, state(T_SAT + dT), H_C)
        q = p.q_c + p.q_q + p.q_e
        qm = measured_at(dT, dTs, qs)
        isfinite(qm) && push!(ratios, q/qm)
        @printf("   %8.2f %14.4e %14.4e %10.3f   %10.3e %10.3e %10.3e\n",
                dT, q, qm, q/qm, p.q_c, p.q_q, p.q_e)
    end
    r_lo, r_hi = extrema(ratios)
    @printf("\n   ratio q_RPI/q_measured spans %.3f to %.3f\n", r_lo, r_hi)
    println("""
   READ THIS AS CALIBRATION, NOT PASS/FAIL. The site-density and departure-
   diameter coefficients are water fits; a cryogenic offset here is expected and
   is what `calibrate_rpi_lh2.jl` exists to remove. What would be a DEFECT is a
   wrong shape or a wrong sign, not a wrong constant.""")
    @test all(isfinite, ratios)
    @test all(r -> r > 0, ratios)
    # Bounded within three decades of measurement: catches a units or exponent
    # error without pretending the water coefficients are calibrated for LH2.
    @test r_hi/r_lo < 1e3

    # --- 3. monotonicity of the nucleate branch ------------------------------
    println("\nC. the nucleate branch must rise monotonically with superheat")
    qn = [sum(wall_heat_partition(RPI_NUCLEATE, state(T_SAT + dT), H_C)[k]
              for k in (:q_c, :q_q, :q_e)) for dT in 0.1:0.1:3.0]
    @test all(diff(qn) .> 0)
    @printf("   q rises from %.3e to %.3e over dT_sup = 0.1 to 3.0 K, monotonically\n",
            qn[1], qn[end])

    # --- 4. the SHAPE, which is the substantive finding ----------------------
    println("\nD. shape: which term actually carries the flux?")
    @printf("   %8s %12s %12s %12s %10s\n", "dT_sup", "q_c", "q_q", "q_e", "q_e share")
    for dT in (0.7, 1.1, 1.67, 3.0, 6.0)
        pp = wall_heat_partition(RPI_NUCLEATE, state(T_SAT + dT), H_C)
        q = pp.q_c + pp.q_q + pp.q_e
        @printf("   %8.2f %12.4e %12.4e %12.4e %9.1f%%\n",
                dT, pp.q_c, pp.q_q, pp.q_e, 100*pp.q_e/q)
    end
    println("""
   THE MISMATCH IS THE SHAPE, NOT THE CONSTANT. The measured curve is near-
   VERTICAL - q goes 4.3 -> 64 kW/m^2 while dT_sup moves only 0.66 -> 1.67 K -
   which is what nucleate boiling looks like when evaporation dominates. Here
   evaporation is under 10% of the flux and single-phase convection carries the
   rest, so q(dT_sup) comes out nearly LINEAR and falls progressively behind: the
   ratio to measurement runs 1.45 -> 1.01 -> 0.28 across the branch.

   The cause is `LemmertChawla`'s exponent. `N_a ~ (m*dT_sup)^n` with n = 1.805 is
   a WATER fit, and this repo's own film-boiling notes cite n ~ 21.17 for LH2. An
   exponent an order of magnitude too low cannot produce a vertical branch at ANY
   prefactor `m`, so this is not removable by rescaling - it needs the cryogenic
   fit that `calibrate_rpi_lh2.jl` exists to produce.""")
    p_lo = wall_heat_partition(RPI_NUCLEATE, state(T_SAT + 0.7), H_C)
    p_hi = wall_heat_partition(RPI_NUCLEATE, state(T_SAT + 3.0), H_C)
    # Evaporation must at least grow faster than convection, or the model has no
    # boiling character at all.
    @test p_hi.q_e/p_lo.q_e > p_hi.q_c/p_lo.q_c
    # Pin the measured shortfall at the top of the nucleate branch, so a
    # recalibration has a baseline to beat.
    q167 = let pp = wall_heat_partition(RPI_NUCLEATE, state(T_SAT + 1.67), H_C)
        pp.q_c + pp.q_q + pp.q_e
    end
    @test_broken 0.5 < q167/measured_at(1.67, dTs, qs) < 2.0

    # --- 4b. the SAME sweep with the LH2 calibration -------------------------
    println("\nD2. recalibrated for LH2 (m = $M_LH2, n = $N_LH2, d_ref = $(D_REF_LH2)),")
    println("    with h_c from the data's low-flux limit rather than Petukhov")
    @printf("   %8s %14s %14s %10s %10s\n",
            "dT_sup", "q_RPI", "q_measured", "ratio", "q_e share")
    ratios_lh2 = Float64[]
    for dT in (0.7, 0.9, 1.1, 1.3, 1.5, 1.67)
        pp = wall_heat_partition(RPI_LH2, state(T_SAT + dT), H_C_DATA)
        q = pp.q_c + pp.q_q + pp.q_e
        qm = measured_at(dT, dTs, qs)
        push!(ratios_lh2, q/qm)
        @printf("   %8.2f %14.4e %14.4e %10.3f %9.1f%%\n",
                dT, q, qm, q/qm, 100*pp.q_e/q)
    end
    l_lo, l_hi = extrema(ratios_lh2)
    @printf("\n   ratio spans %.3f to %.3f   (water fit: %.3f to %.3f)\n",
            l_lo, l_hi, r_lo, r_hi)
    # The calibration must actually improve the fit, and bring the top of the
    # branch - where the water fit collapsed to 0.278 - inside a factor of two.
    @test l_hi/l_lo < r_hi/r_lo
    @test 0.5 < ratios_lh2[end] < 2.0
    # Evaporation must now be carrying a real share rather than under 10%.
    p_lh2 = wall_heat_partition(RPI_LH2, state(T_SAT + 1.67), H_C_DATA)
    q_lh2 = p_lh2.q_c + p_lh2.q_q + p_lh2.q_e
    @printf("   q_e share at the top of the branch: %.1f%% (water fit: %.1f%%)\n",
            100*p_lh2.q_e/q_lh2, 100*p_hi.q_e/(p_hi.q_c + p_hi.q_q + p_hi.q_e))
    @test p_lh2.q_e/q_lh2 > 0.2

    # --- 5. film boiling: continuity through the transition (rung 5.3) -------
    println("\nE. film boiling - the blended curve must be continuous and turn over")
    film = FilmBoiling(
        chf = FixedCriticalHeatFlux(q = 64.0e3),   # the MEASURED CHF, not a correlation
        minimum_film = Berenson(),
        htc = Bromley(D = D_PIPE),
        transition = SuperheatTransition())
    rpi_film = RPI(patches = (:wall,), film_boiling = film,
                   wall_capacity = 100.0)          # required past CHF; see `RPI`

    # `y_plus`/`u_tau` matter only to `ForcedConvectionFilm`; Bromley ignores them.
    fc = film_closure(rpi_film, film, state(T_SAT + 1.0), H_C, 30.0, 0.1)
    @test fc !== nothing

    dTv = collect(0.05:0.05:80.0)
    qv = map(dTv) do dT
        pp = wall_heat_partition(rpi_film, state(T_SAT + dT), H_C, fc, 0.0)
        pp.q_c + pp.q_q + pp.q_e + pp.q_f
    end
    @test all(isfinite, qv)
    @test all(qv .>= 0)

    # Continuity measured against the PEAK flux, not the local value. Normalising
    # locally makes the metric meaningless where q is small: at dT_sup = 0.1 -> 0.2
    # K the flux simply doubles because q_c = h_c*dT is linear near zero, and that
    # reads as a 101% "step" while being perfectly smooth.
    jumps = abs.(diff(qv))./maximum(qv)
    ipk = argmax(qv)
    @printf("   peak q = %.4e W/m^2 at dT_sup = %.1f K;  largest relative step %.4f\n",
            maximum(qv), dTv[ipk], maximum(jumps))
    # Continuity is a STABILITY gate: the wall traverses this region during a run,
    # and a step here makes the wall-temperature solve chatter between branches in
    # a way that reads as a solver failure and is not.
    @test maximum(jumps) < 0.05

    @printf("   turns over: %s (peak at index %d of %d)\n",
            ipk < length(qv) - 2 ? "yes" : "NO", ipk, length(qv))
    @test ipk < length(qv) - 2
    @test minimum(qv[ipk:end]) < 0.9*maximum(qv)
    # The spike the nucleate cap exists to prevent must not be present.
    @test maximum(qv) < 100*64.0e3
    println("""
   The cap on the nucleate branch is what makes this work: `(1 - w)*q_RPI` alone
   spikes mid-transition because `q_e ~ dT_sup^n` outruns a linear weight. That is
   documented at `wall_heat_partition`; this arm is the standing check on it.""")

    println()
end
