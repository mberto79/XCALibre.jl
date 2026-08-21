# =============================================================================
#  Rung 3.1 - 0D relaxation to saturation
# =============================================================================
#
#  Plan: dev_notes_LH2_validation_plan.md, Stage 3. FIRST rung with vapour.
#
#  Rung 2.9 verified the temperature field a phase-change model will read. This
#  is the first rung that acts on it, and it is deliberately the smallest thing
#  that can: a uniform box, no flow, no gradients, no wall boiling. One rate law,
#  one sink, one exact answer.
#
#  THE EXACT ANSWER
#
#  With no flow and no conduction the energy equation is
#
#      rho_cp * dT/dt = -mdot*h_fg
#
#  and the Lee rate is linear in the superheat,
#
#      mdot = C*(T - T_sat),   C = r*alpha_l*rho_l/T_sat
#
#  with `r` the prescribed Lee relaxation coefficient [1/s]. There is no
#  interfacial-area factor: `r` is already volumetric (`uses_interfacial_area`).
#
#  so the superheat decays exponentially,
#
#      T(t) - T_sat = (T_0 - T_sat)*exp(-t/tau),   tau = rho_cp/(C*h_fg)
#
#  TWO INDEPENDENT CHECKS, and they fail differently
#
#  1. RATE. `tau` measured against the expression above. This tests the rate law,
#     the alpha/density weighting and the latent sink together - everything that
#     sets HOW FAST.
#
#  2. ENERGY. The vapour mass created must equal `rho_cp*(T_0 - T_end)/h_fg`.
#     This is conservation and holds whatever `C` is, so it stays valid even
#     where the linearisation does not. A model that gets the rate wrong but the
#     energy right is mis-calibrated; one that gets the energy wrong is losing
#     mass or heat, which is far more serious.
#
#  WHY NOT EQUAL DENSITIES
#
#  The obvious way to isolate the rate law would be `rho_l = rho_v`, killing the
#  volume creation. It is not done here because the density ratio is precisely
#  what makes the two branches asymmetric, and hiding it would hide the model's
#  actual behaviour. The volume created is therefore real, and the box needs
#  somewhere to put it - hence a COMPRESSIBLE vapour.
#
#  `ConstantSaturation` then freezes `T_sat` regardless of what the pressure
#  does, which decouples the driving force from the pressure response entirely.
#  Without it a rising pressure would raise `T_sat`, the superheat would decay for
#  two reasons at once, and neither could be attributed. (The pipe case already
#  uses `ConstantSaturation` for its own reasons.)
#
#  The superheat is kept small (0.2 K) so `alpha` and `rho_cp` barely move and the
#  linearisation is clean. Check 2 is then repeated at 5 K, where it is not.
#
#  THE CONVENTION ARM - what this rung was built to settle, and did
#
#  `phase_change_rate!` used to pass the TRACKED `alpha` into the Lee weight,
#  while the model documents the LIQUID fraction and
#  `multiphase_liquid_phase` states that "the phase-change sink, its sign, ..."
#  must use `liquid_phase`. Those agree only while `alpha` tracks the liquid -
#  the default, and NOT the LH2 pipe's configuration, which tracks the vapour
#  because the dilute phase should be tracked.
#
#  Arm C runs the identical physical state with the phases swapped. Measured
#  BEFORE the fix: tau = 25.0 s against 4.02 s, a factor of 6.2. AFTER passing
#  `alpha_liq`: 0.80%. The arm is kept as a regression guard.
# =============================================================================

using XCALibre
using Test
using Printf
using LinearAlgebra

const GRIDS = pkgdir(XCALibre, "examples/0_GRIDS")

# --- state: saturated LH2 at 0.4 MPa ----------------------------------------
const P_SAT  = 0.4e6
const P_TAB  = (0.30e6, 1.25e6)
const SATC   = build_saturation_curve(H2(), p=P_TAB, T=(19.0, 120.0), np=201, nT=201)
const T_SAT  = saturation_temperature(SATC, P_SAT)
const H_FG   = latent_heat(SATC, P_SAT, 0.0)
const SAT_FROZEN = ConstantSaturation(SATC, P_SAT)

# Tabulated property values at the saturation point. Held as constants rather
# than as live tables: this rung is about the RATE LAW, and a varying cp would
# make `tau` a moving target for no benefit. Rung 3.2 uses the live tables.
const LIQ = RealFluid(H2(), :liquid;  p=P_TAB, T=(20.0, 32.5), np=81, nT=81, verbose=false)
const VAP = RealFluid(H2(), :vapour;  p=P_TAB, T=(20.0, 34.0), np=81, nT=81, verbose=false)
const RHO_L = table_lookup(LIQ.rho.rho, P_SAT, T_SAT)
const RHO_V = table_lookup(VAP.rho.rho, P_SAT, T_SAT)
const CP_L  = table_lookup(LIQ.cp.cp,   P_SAT, T_SAT)
const CP_V  = table_lookup(VAP.cp.cp,   P_SAT, T_SAT)
const K_L   = table_lookup(LIQ.k.k,     P_SAT, T_SAT)
const MU_L  = table_lookup(LIQ.mu.mu,   P_SAT, T_SAT)
const R_SP  = 8.314462618/2.01588e-3       # H2 specific gas constant [J/kg/K]

const ALPHA_L0 = 0.9        # liquid fraction
const DIAM     = 1.0e-3
const LEE_R    = 0.3        # Lee relaxation coefficient [1/s]

# --- the analytic solution ---------------------------------------------------
rho_cp(a_l) = a_l*RHO_L*CP_L + (1 - a_l)*RHO_V*CP_V

"C in mdot = C*(T - T_sat), for evaporation. `r` is volumetric, so there is no
interfacial-area factor - see `uses_interfacial_area`."
lee_C(a_l) = LEE_R*a_l*RHO_L/T_SAT

"Analytic relaxation time of the superheat."
tau_exact(a_l) = rho_cp(a_l)/(lee_C(a_l)*H_FG)

const RTOL = 1.0e-10
const SOLVERS = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-8, itmax=2000),
    p_rgh = SolverSetup(solver=Bicgstab(), preconditioner=DILU(),
                        convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-8, itmax=5000),
    alpha = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                        convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-10, itmax=1500),
    T = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-8, itmax=2500),
)
const SCHEMES = (
    U     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    p     = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
    p_rgh = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
    alpha = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    T     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
)

"""
    run_0d(; dT0, track_vapour, dt, iterations, nsample)

Uniform sealed box. `track_vapour = false` puts the liquid first, so `alpha`
measures the liquid fraction (the solver default). `track_vapour = true` swaps
them and sets `liquid_phase = 2`, which is the LH2 pipe's configuration.

Returns the sampled superheat history, so the decay can be fitted rather than
inferred from the endpoints.
"""
function run_0d(; dT0, track_vapour=false, dt=1.0e-2, iterations=1400, nsample=28)
    mesh = UNV2D_mesh(joinpath(GRIDS, "quad.unv"), scale=1.0e-4)   # 0.1 m box
    backend = CPU(); hardware = Hardware(backend=backend, workgroup=AutoTune())
    mesh_dev = adapt(backend, mesh)

    liquid = Phase(rho=ConstEos(rho=RHO_L), mu=MU_L, k=K_L, cp=CP_L, beta=0.0164)
    # Compressible, so the volume created by evaporation has somewhere to go in a
    # sealed box. `ConstantSaturation` keeps T_sat fixed regardless.
    vapour = Phase(rho=IdealGas(R=R_SP), mu=1.11e-6, k=0.0169, cp=CP_V)

    phases = track_vapour ? (vapour, liquid) : (liquid, vapour)
    liquid_index = track_vapour ? 2 : 1
    alpha0 = track_vapour ? (1 - ALPHA_L0) : ALPHA_L0

    model = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = Mixture(diameter=DIAM),
            phases = phases,
            liquid_phase = liquid_index,
            phase_change = Lee(r=LEE_R),
            interfacial_area = DispersedBubbles(diameter=DIAM),
            saturation = SAT_FROZEN,
            h_fg = H_FG,
            p_operating = P_SAT,
            # PRESSURE WORK OFF, and this is not optional here. Evaporation in a
            # SEALED box creates volume, the box pressurises, and `S_T =
            # beta*T*dp/dt` then HEATS the fluid - so the superheat is driven by
            # two competing terms and the exponential relaxation is not the
            # answer to any question this rung is asking.
            #
            # It was measured before being switched off: at 0.2 K superheat the
            # pressure work WON and the superheat grew (fitted tau = -27.8 s),
            # while at 5 K the latent sink dominated and it relaxed but produced
            # 115% of the vapour the sensible heat could account for. Both are
            # real physics of a sealed box, and neither is the rate law.
            #
            # Rungs 2.5 and 2.9 already verify the pressure-work path. Isolating
            # it out here is the same discipline as everywhere else in the ladder.
            pressure_work_relax = 0.0,
            gravity = Gravity([0.0, 0.0, 0.0])),
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=T_SAT),
        domain = mesh_dev)

    walls(f) = [f(:inlet), f(:outlet), f(:bottom), f(:top)]
    noSlip = [0.0, 0.0, 0.0]
    BCs = assign(region = mesh_dev, (
        U = [Wall(:inlet, noSlip), Wall(:outlet, noSlip),
             Wall(:bottom, noSlip), Wall(:top, noSlip)],
        p_rgh = walls(Zerogradient),
        alpha = walls(Zerogradient),
        T = walls(Zerogradient),              # adiabatic: the only sink is latent
    ))

    initialise!(model.momentum.U, noSlip)
    initialise!(model.fluid.p_rgh, 0.0)
    initialise!(model.fluid.alpha, alpha0)
    initialise!(model.energy.T, T_SAT + dT0)

    chunk = max(iterations ÷ nsample, 1)
    ts = Float64[]; dTs = Float64[]
    vap_mass(m) = begin
        a = m.fluid.alpha.values
        av = track_vapour ? a : (1 .- a)
        rv = m.fluid.phases[track_vapour ? 1 : 2].rho
        rvv = rv isa ConstantScalar ? fill(rv.values, length(a)) : rv.values
        sum(av .* rvv .* [c.volume for c in mesh.cells])
    end
    a_l0 = track_vapour ? 1 - alpha0 : alpha0

    cfg(n) = Configuration(solvers=SOLVERS, schemes=SCHEMES,
        runtime=Runtime(iterations=n, time_step=dt, write_interval=-1),
        hardware=hardware, boundaries=BCs)

    # BASELINE AFTER ONE STEP, and this matters. Before the first `run!`,
    # `update_phase_state!` has not executed and the vapour density field is still
    # all zeros, so `vap_mass` returns 0. Differencing from there reports the
    # vapour that was ALWAYS PRESENT as though it had just been created - which is
    # exactly what produced the "+76% energy imbalance" first recorded here. The
    # superheat at the baseline is captured with it, so the energy expectation is
    # measured over the same interval.
    run!(model, cfg(1))
    m0 = vap_mass(model)
    dT_ref = sum(model.energy.T.values)/length(model.energy.T.values) - T_SAT
    done = 1
    while done < iterations
        n = min(chunk, iterations - done)
        run!(model, cfg(n))
        done += n
        push!(ts, done*dt)
        push!(dTs, sum(model.energy.T.values)/length(model.energy.T.values) - T_SAT)
        all(isfinite, model.energy.T.values) || break
    end

    T_end = sum(model.energy.T.values)/length(model.energy.T.values)
    return (ok = all(isfinite, model.energy.T.values),
            ts = ts, dTs = dTs, dT0 = dT0, dT_ref = dT_ref,
            T_end = T_end, dT_end = T_end - T_SAT,
            dmass = vap_mass(model) - m0,
            volume = sum(c.volume for c in mesh.cells),
            a_l0 = a_l0,
            alpha_end = sum(model.fluid.alpha.values)/length(model.fluid.alpha.values))
end

"Least-squares tau from log(superheat) against t, over the samples that are still
well above round-off."
function fit_tau(ts, dTs)
    keep = findall(i -> dTs[i] > 1e-4*dTs[1] && dTs[i] > 0, eachindex(dTs))
    length(keep) >= 3 || return NaN
    t = ts[keep]; y = log.(dTs[keep])
    slope = (hcat(ones(length(t)), t) \ y)[2]
    return -1/slope
end

@testset "3.1 zero-dimensional relaxation to saturation" begin

    println("\n", "="^80)
    println(" Rung 3.1 - 0D relaxation to saturation (Lee, dispersed bubbles)")
    println("="^80)
    @printf("\n  saturated LH2 at %.2f MPa:  T_sat = %.3f K   h_fg = %.1f kJ/kg\n",
            P_SAT/1e6, T_SAT, H_FG/1e3)
    @printf("  rho_l = %.3f   rho_v = %.3f   cp_l = %.0f   cp_v = %.0f\n",
            RHO_L, RHO_V, CP_L, CP_V)
    @printf("  alpha_l = %.2f   d = %.1e m   Lee r = %.3g 1/s\n", ALPHA_L0, DIAM, LEE_R)
    @printf("  analytic tau = %.4f s\n\n", tau_exact(ALPHA_L0))

    # --- A. evaporation, alpha tracks the liquid (solver default) ------------
    println("A. evaporation, alpha tracks the LIQUID (default convention)")
    monitor_linear_solves!()
    a = run_0d(dT0 = 0.2)
    a_g1 = check_linear_convergence(verbose=false)
    tau_a = fit_tau(a.ts, a.dTs)
    @printf("   tau measured %.4f s  vs analytic %.4f s  (%+6.2f%%)   G1 %s\n",
            tau_a, tau_exact(ALPHA_L0), 100*(tau_a/tau_exact(ALPHA_L0) - 1),
            a_g1 ? "ok" : "FAIL")
    @test a.ok
    @test a_g1
    @test a.dT_end < a.dT0            # it must actually relax
    @test isfinite(tau_a)
    @test abs(tau_a/tau_exact(ALPHA_L0) - 1) < 0.10

    # Energy: the vapour created must account for the sensible heat released.
    expected = rho_cp(a.a_l0)*(a.dT_ref - a.dT_end)*a.volume/H_FG
    @printf("   vapour mass created %.6e kg  vs energy balance %.6e kg  (%+6.2f%%)\n",
            a.dmass, expected, 100*(a.dmass/expected - 1))
    # A REAL but MODEST residual, ~4%. The +77% first recorded here was a
    # measurement error, since fixed: the baseline was taken before the first
    # step, when `update_phase_state!` had not run and the vapour density field
    # was still zero, so the vapour that was always present was counted as newly
    # created.
    #
    # What survives, measured by sweeping the superheat so the linearisation in
    # `expected` (constant rho_cp at the INITIAL alpha) is squeezed out:
    #
    #     dT0 = 0.02 K -> +4.12%      1.0 K -> +12.40%
    #           0.05 K -> +4.48%      5.0 K -> +23.77%
    #           0.20 K -> +6.16%
    #
    # It does NOT vanish as dT0 -> 0, so ~4% is genuine; the growth above that is
    # `expected` crediting more sensible heat than the shrinking liquid holds.
    #
    # NOT the alpha clamp: `alpha` runs 0.900 -> 0.892 here and never approaches
    # a bound, so `clamp(a, 0, 1)` never fires. That hypothesis is dead.
    @test_broken abs(a.dmass/expected - 1) < 0.01

    # --- B. condensation: the same law with the sign reversed ----------------
    println("\nB. condensation (subcooled), same convention")
    b = run_0d(dT0 = -0.2)
    tau_b = fit_tau(b.ts, -b.dTs)
    @printf("   subcooling %.4f -> %.4f K   tau measured %.1f s   vapour change %+.3e kg\n",
            -b.dT0, -b.dT_end, tau_b, b.dmass)
    @test b.ok
    @test b.dT_end > b.dT0            # must relax UP towards saturation
    # The condensation weight is (1-alpha)*rho_v = 0.484 against evaporation's
    # alpha*rho_l = 56.7 - a factor of 117, so this branch is ~117x slower BY
    # CONSTRUCTION and the measured 694 s against A's 4.0 s is the model's own
    # asymmetry, not an error. What is tested here is the SIGN and direction.
    # Condensation consumes vapour, correctly. The apparent "+3.6e-3 kg created"
    # first recorded here was the same zero-baseline error as arm A.
    @test b.dmass < 0

    # --- C. the convention arm ----------------------------------------------
    println("\nC. identical physical state, alpha tracks the VAPOUR (pipe convention)")
    println("   `liquid_phase = 2`. Same physics, so tau must be unchanged.")
    c = run_0d(dT0 = 0.2, track_vapour = true)
    tau_c = fit_tau(c.ts, c.dTs)
    @printf("   tau measured %.4f s  vs arm A %.4f s  (%+8.2f%%)\n",
            tau_c, tau_a, 100*(tau_c/tau_a - 1))
    @test c.ok
    if isfinite(tau_c) && abs(tau_c/tau_a - 1) < 0.10
        println("   -> conventions agree; the tracked/liquid distinction is handled.")
    else
        println("   -> THEY DISAGREE. `phase_change_rate!` passes the TRACKED alpha into")
        println("      a weight the Lee model documents as the LIQUID fraction, and")
        println("      `multiphase_liquid_phase` states the phase-change sink must use")
        println("      `liquid_phase`. This is the configuration the LH2 pipe runs in.")
    end
    # FIXED 2026-08-19: `phase_change_rate!` now passes `alpha_liq`, and `Lee`
    # takes a prescribed volumetric `r` instead of an area-scaled derived beta.
    # Before: 25.0 s against 4.02 s, a factor of 6.2. After: 0.80%.
    @test isfinite(tau_c)
    @test abs(tau_c/tau_a - 1) < 0.05

    # --- D. energy conservation beyond the linear regime ---------------------
    println("\nD. energy balance at 5 K superheat (linearisation no longer valid)")
    d = run_0d(dT0 = 5.0)
    expected_d = rho_cp(d.a_l0)*(d.dT_ref - d.dT_end)*d.volume/H_FG
    @printf("   superheat %.3f -> %.3f K   vapour %.6e kg vs %.6e kg (%+6.2f%%)\n",
            d.dT0, d.dT_end, d.dmass, expected_d, 100*(d.dmass/expected_d - 1))
    @test d.ok
    # 14 s is ~3.4 tau, so exp(-3.4) = 0.033 of 5 K is ~0.17 K; alpha moves
    # appreciably at this superheat and slows it further. Below 10% is the
    # defensible statement for this run length.
    @test d.dT_end < 0.10*d.dT0
    # +23.8%, most of which is `expected_d` and not the solver: it holds rho_cp
    # at the initial alpha, which falls from 0.900 to 0.696 over this run.
    @test_broken abs(d.dmass/expected_d - 1) < 0.05

    println()
end
