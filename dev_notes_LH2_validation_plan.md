# LH2 boiling model — incremental verification & validation plan

Companion to [`dev_notes_LH2_pipe_boiling.md`](dev_notes_LH2_pipe_boiling.md)
(what was built and what failed), [`dev_notes_LH2_tanks.md`](dev_notes_LH2_tanks.md)
and [`dev_notes_LH2_implementation_plan.md`](dev_notes_LH2_implementation_plan.md).

**Target:** Tatsumoto et al. (2014), forced-convection boiling of saturated LH2
in vertical heated pipes — full boiling curve through nucleate, transition and
film boiling.

---

## Why start over from the bottom

The debugging record in `dev_notes_LH2_pipe_boiling.md` is unusually thorough and
still produced "several confident diagnoses that were later disproved". That is
not a failure of care. It is the predictable outcome of **single-variable
elimination on a case with more than one independent defect**.

At least four are documented, each individually sufficient to break the case:

| # | Defect | Level |
|---|---|---|
| 1 | Pressure linear solve returning unconverged iterates (45 of 45 at `itmax`) | numerics |
| 2 | Explicit thermo-acoustic loop run ~25x over its CFL limit | time integration |
| 3 | Volume-form pressure equation not enforcing mass conservation | formulation |
| 4 | Discrete vapour mass balance violated ~100% by the alpha update form | formulation |

Turn one off and three remain, so the case still diverges and the term looks
exonerated. Every elimination in that document is therefore sound as a *relative*
comparison and unsound as an *absolute* one — which is exactly what the notes
themselves conclude.

**The governing principle for this plan:** every case must be small enough that
only one thing can be wrong with it, and must have an answer known independently
of the code.

A second discipline point, taken straight from the existing notes: *stop drawing
stability conclusions from 15–60 step runs.* DILU looked conclusive at 15 steps
and did not hold. Any claim of stability states the number of steps and the
physical time reached, or it is written as "not yet diverging at N steps".

---

## Standing acceptance gates

These apply at **every** rung. A case that passes its physics check but fails a
gate has not passed.

| Gate | Statement | How measured |
|---|---|---|
| **G1 Linear convergence** | No solve returns at `itmax`; every solve reaches `atol + rtol*norm(b)` | log `Krylov.iteration_count` and the returned residual per call; report `n_at_itmax` |
| **G2 Conservation** | Per-phase mass balances inflow/outflow/storage; global energy budget closes | `energy_budget!`; a new per-phase mass audit |
| **G3 Step independence** | Halving `dt` moves the answer by less than the acceptance tolerance | run each case at `dt` and `dt/2` |
| **G4 Grid independence** | Three mesh levels, observed order consistent with the scheme | required wherever a steady answer is claimed |

G1 is a **precondition, not a result**. Until it holds, "diverged" and "the
pressure was never solved" are indistinguishable — which is what happened.

---

## Prerequisite work (blocks Stages 4–5, do early)

1. **Expose the wall-boiling diagnostics.** `WallBoilingState` carries `T_wall`,
   `q_evap`, `q_quench`, `q_conv` per face and is local to `MULTIPHASE`. No
   boiling-curve comparison is possible without `T_wall` in the output. Already
   flagged as a known gap; it is now on the critical path.
2. **A G1 reporting mode** on the solver — per-solve iteration count, final
   residual, count of solves hitting `itmax`. Cheap, and it retires an entire
   class of false conclusion.
3. **A per-phase mass audit** to sit alongside `energy_budget!`, with the same
   design rule that file already states: validate the instrument on a case whose
   answer is known before trusting it anywhere else.

---

## Stage 0 — Numerics floor (single phase, no boiling, no phase change)

Nothing on this rung is about hydrogen. It is about whether the discretisation on
*this mesh* with *this linear solver* reproduces answers that are not in dispute.

| # | Case | Known answer | Acceptance |
|---|---|---|---|
| 0.1 | **Linear-solver gate** applied to every existing multiphase case | — | 0 solves at `itmax` |
| 0.2 | **Uniform-flow preservation**, O-grid, uniform properties, no gravity | nothing changes, ever | drift < 1e-12 over 1000 steps |
| 0.3 | **Hydrostatic, uniform rho** (exists: `2d_multiphase_hydrostatic.jl`) | `U = 0`, `p = rho·g·h` | `max\|U\|` at machine zero |
| 0.4 | **Hydrostatic, VARIABLE rho** — 1D column, stratified or `rho = rho(T)` | `U = 0`, `p = ∫rho·g·dz` | `max\|U\|` at machine zero |
| 0.5 | **Laminar Poiseuille**, pipe O-grid | parabolic profile, `f = 64/Re` | `f` within 1%, profile within 1% |
| 0.6 | **Turbulent pipe** (exists: `3d_LH2_pipe_singlephase_validation.jl`) | Petukhov `f`; `u_tau = U·sqrt(f/8)`; log law | both routes agree within 5% |
| 0.7 | **Heated turbulent pipe** | Dittus–Boelter `Nu = 0.023 Re^0.8 Pr^0.4` | within 15% |
| 0.8 | **Grid convergence** on 0.5 and 0.6 | observed order ≈ scheme order | 3 levels, monotone |
| 0.9 | **Step independence** on 0.6 | steady answer independent of `dt` | G3 |

### 0.4 is the highest-value cheap test in this document

`dev_notes_LH2_pipe_boiling.md` ends with variable-`rho` buoyancy as the
**leading unfalsified suspect** — `phi_gf!` and `well_balanced_pressure_grad!`
build the buoyancy from `snGrad(rho)` and are well balanced by construction only
for piecewise-constant `rho`. There is currently **no test of this**, and it can
be settled on a 1D column of a few hundred cells in an afternoon. If a stratified
static column generates spurious velocity, the hypothesis is confirmed and the
fix is localised. If it does not, the leading suspect is dead and 2.5 below
becomes the next discriminator.

Do 0.1 and 0.4 first, before anything else in this plan.

### Why 0.6 and 0.7 are not optional

`h_c = rho·cp·u_tau / T+` in the RPI convective leg is directly proportional to
the friction velocity the wall function produces. A measured `h_conv` of
3692–5388 W/m²/K against a Dittus–Boelter estimate of ~10,200 has already been
observed. If `u_tau` is 2–3x low, `q_conv` is starved and the partition makes up
the difference through evaporation — inflating the wall superheat and corrupting
every boiling-curve comparison downstream. The notes call this "the single most
valuable first validation" and it is still open.

---

## Stage 1 — Adiabatic dispersed bubbly flow

No heat transfer, no phase change. Purely: does the drift-flux mixture transport
two phases correctly?

| # | Case | Known answer | Tests |
|---|---|---|---|
| 1.1 | **Zero-slip degeneracy** — `rho1 = rho2`, `mu1 = mu2`, prescribed uniform U | alpha translates exactly; `∫alpha·dV` constant | MULES boundedness, alpha conservation, numerical diffusion |
| 1.2 | **Drift velocity vs terminal velocity** (unit-level, no CFD) | `U_dm` from the Schiller–Naumann force balance, iterated in `Re_p` | `compute_Ur!` implementation |
| 1.3 | **Rising bubble swarm** in a still column | front rises at 1.2's velocity | drift flux in the alpha eqn + momentum slip stress, consistently |
| 1.4 | **Batch sedimentation (Kynch kinematic wave)** | exact front velocities from the drift-flux curve `j_d(alpha)` | the *coupled* alpha–drift system |
| 1.5 | **Steady bubbly pipe flow** | Zuber–Findlay `alpha_g = j_g/(C0·j + u_gj)` | area-averaged void in the target topology |
| 1.6 | **Adiabatic two-phase pressure drop** | homogeneous model and Lockhart–Martinelli | mixture momentum, two-phase multiplier |
| 1.7 | **Per-phase mass audit** (standing) | closed domain: both phase masses constant | G2 |

### 1.1 is the cleanest test of the alpha machinery and it does not exist

With equal densities `Ur ≡ 0` identically, so alpha becomes a passive scalar in a
prescribed field and the exact answer is pure translation. It isolates MULES, the
boundedness argument and alpha conservation from *everything else*. Given that
all three attempts at reformulating alpha transport (conservative explicit,
implicit, mass-form) made the measured balance worse, this is where that
reformulation should be characterised — not on the coupled boiling case.

Measure and record the **numerical diffusion** here too: how far a step in alpha
smears over N steps sets a floor on how sharp any predicted void profile can ever
be, and it is a number you will want when a void profile comes out too flat.

### 1.4 is the missing analytic test for this stage

Kynch's kinematic-wave solution for batch sedimentation gives closed-form front
velocities for the coupled volume-fraction/drift system in 1D — not just the
drift closure in isolation. It is the standard verification case for drift-flux
codes and there is nothing equivalent in the repo.

### A modelling caveat to establish now, not later

Manninen drift-flux with Schiller–Naumann assumes rigid spheres. Real hydrogen
bubbles at the 1 mm default `diameter` are wobbling ellipsoids. Compare 1.2's
value against the Grace/Clift regime diagram and **write down the error band** —
otherwise it will later be attributed to the boiling model.

Two related concerns are already flagged in the source: `tau_d` is built from the
vapour viscosity, and `tau_d·|grad u| ~ 880` near the wall against the `<< 1` the
local-equilibrium assumption requires. The second means the drift-flux assumption
is **violated in the near-wall cells** — precisely where the boiling happens.
Quantify it here, because it bounds what Stage 4 can ever achieve.

---

## Stage 2 — Variable equation of state

| # | Case | Known answer | Tests |
|---|---|---|---|
| 2.1 | **Table vs direct EOS** (exists, 124 tests) + **derivative consistency** | tabulated `psi`, `beta` must match finite differences of tabulated `rho` | table self-consistency |
| 2.2 | **Acoustic wave in a sealed duct** | `c = 1/sqrt(rho·psi)`, period `2L/c` | correct wave speed; **calibrates the `dt` limit** |
| 2.3 | **Sealed isochoric heating** (exists: `unit_test_compressible_ullage.jl`) | `dp/dt = R·Q/(V·cv)` | compressible pressure path |
| 2.4 | **Isothermal compression of a sealed ullage** by a moving liquid level | `p2` from the EOS at the new `rho` | alpha/pressure coupling, no phase change |
| 2.5 | **Compressible single-phase through-flow duct** | `dp/dx = f·rho·U²/(2D)`, density nearly constant | **the decisive test** — see below |
| 2.6 | **Well-balancedness with `rho = rho(p,T)`** | 0.4 repeated with a real tabulated EOS | the §5.4 concern in its real form |

### 2.5 separates the two surviving hypotheses

The failure is known to be specific to the compressible branch, and to occur with
no phase change and no heating. Two explanations remain: an ill-conditioned
pressure solve on a 44:1 aspect-ratio O-grid, or a defect in the compressible
formulation itself.

Run it on a **straight duct with a benign mesh and a verified-converged linear
solve** (G1 enforced):

- **stable** → the formulation is sound; O-grid conditioning is the problem, and
  the work is preconditioning/mesh, not physics.
- **diverges** → the formulation is wrong, and it is now wrong on a case small
  enough to instrument cell by cell.

There is currently no way to make that split, and all of Stages 3–5 sits on it.

### 2.2 converts "root cause (1)" from a hypothesis into a number

The thermo-acoustic loop analysis gives `dt < dx/c` with `c ~ 420` m/s, i.e.
`dt < 8e-8` s against the `2e-6` the case uses. That is currently a paper
estimate. Measuring the propagation speed of a small perturbation directly
confirms the code carries the right `c`, and yields a **defensible `dt` rule** to
apply at every later rung rather than tuning `dt` per case and hoping.

The constraint **tightens on the film-boiling branch**, where the near-wall cell
is vapour; re-derive it at 5.4.

### 2.1's derivative check matters more than it looks

The pressure equation uses `psi`; the mass balance uses `rho`. If tabulated `psi`
is not the exact derivative of tabulated `rho` on the same bilinear grid, the two
disagree by a fixed amount and the result is a mass-conservation error no solver
work can remove. Same argument for `beta` against `drho/dT`. Cheap, and
unchecked.

---

## Stage 3 — Bulk phase change

| # | Case | Known answer | Tests |
|---|---|---|---|
| 3.1 | **0D relaxation to saturation**, both directions | `T → T_sat` exponentially; vapour mass = excess enthalpy/`h_fg` | rate sign, coefficient meaning, latent sink vs alpha source consistency |
| 3.2 | **1D Stefan problem**, both directions | `delta(t) = 2·lambda·sqrt(a·t)`, exact `T` profile | **the standard analytic phase-change verification** |
| 3.3 | **Sucking interface** (Welch & Wilson) | exact, including the dilatation velocity | `add_phase_change_volume!` |
| 3.4 | **Scriven bubble growth** | `R(t) = 2·beta·sqrt(a·t)` | the same physics in the geometry RPI produces |
| 3.5 | **K-Site tank self-pressurisation** (case exists) | NASA TM-103804 data | validation of Schrage / MeJ / Lee |
| 3.6 | **Coupled mass + energy budget** (standing) | closes to solver tolerance | G2 |

### 3.2 is the largest single gap in the plan as you stated it

There is **no analytic phase-change verification anywhere in the repo**. The
Stefan problem is the canonical one, and it exercises the interfacial mass
transfer rate, the latent sink, the alpha source and the volume dilatation
simultaneously against a closed-form answer.

Run it against **both** closures. `Lee` carries a free coefficient, so the
verification statement is specific: as `sigma` increases the solution must
**converge onto the Stefan solution** — the interface-limited regime giving way
to the thermally-limited one. If it converges to something else, the coupling is
wrong. If it never converges, the coefficient is doing work it should not be.

### Run condensation as well as evaporation

Every measurement in the existing notes is evaporative. The `Lee` implementation
switches only the `alpha·rho` weighting between branches while `(T − T_sat)/T_sat`
carries the sign — cheap to get subtly wrong, and a sign error on the
condensation branch will not show up until vapour reaches a cold region, which is
exactly what happens downstream of a boiling wall.

---

## Stage 4 — Wall boiling, nucleate branch

| # | Case | Known answer | Tests |
|---|---|---|---|
| 4.1 | **RPI partition algebra** (exists, 95 tests) | `q_c + q_q + q_e = q_w` | inversion, already at 5e-13 |
| 4.2 | **0D boiling curve from the closures alone**, no CFD | Rohsenow / Kutateladze for LH2; Tatsumoto's nucleate branch | **are the coefficients even in the right decade?** |
| 4.3 | **LH2 pool boiling, flat plate** | Kutateladze / Rohsenow, LH2 data | wall boiling without the flow or turbulence model |
| 4.4 | **Bartolomei subcooled boiling in a tube — WATER** | measured radial/axial void profiles and `T_wall` | **the decisive implementation-vs-coefficients split** |
| 4.5 | **Energy budget with boiling on** | closes | re-measure the 2–7e8 W/m³ spurious source with G1 enforced |
| 4.6 | **Tatsumoto nucleate branch**, below CHF only | digitised `q(T_w)` curve | the target, finally |

### 4.4 is the case I would most strongly add

Bartolomei & Chanturiya (1967) — vertical tube, water, subcooled flow boiling,
with measured **radial and axial void fraction profiles** and wall temperatures
across several pressures and heat fluxes. It is *the* published RPI validation
benchmark; STAR-CCM+, ANSYS Fluent and NEPTUNE_CFD all report against it.

Three reasons it belongs here:

1. **Water properties are known to five digits**, so the property tables stop
   being a variable.
2. **The RPI coefficients are inside their fitted range.** `LemmertChawla`
   (m = 210, n = 1.805) and `TolubinskyKostanchuk` (0.6 mm, 45 K) are
   water-fitted, as the notes state. On water, any disagreement is the
   *implementation*. On LH2 it could be either.
3. **It measures void fraction profiles** — precisely the quantity the alpha
   transport is suspected of getting wrong, in precisely the flow topology of the
   target case.

That gives a split you currently cannot make:

| Bartolomei | Tatsumoto | Conclusion |
|---|---|---|
| passes | fails | the cryogenic coefficients need recalibration — expected, and `calibrate_rpi_lh2.jl` already exists for it |
| fails | fails | the implementation or the transport is wrong; stop tuning coefficients |

### 4.2 before any CFD

Sweeping `T_w` through the partition at a single point costs seconds and tells
you whether water-fitted coefficients put the LH2 nucleate branch within a factor
of ~2 of the measured curve. If they are off by 3x, that would otherwise be
discovered after weeks of solver work. The calibration scripts
(`calibrate_rpi_lh2.jl` and siblings) already exist — this makes them a formal
gate rather than an ad-hoc tool.

---

## Stage 5 — CHF, transition and film boiling

| # | Case | Known answer | Tests |
|---|---|---|---|
| 5.1 | **CHF correlation check, 0D** | Tatsumoto's measured 64 kW/m² | see below |
| 5.2 | **Film boiling correlation check, 0D** | Bromley (cylinder), Berenson (plate) | implementation vs published formula |
| 5.3 | **Transition blending continuity** | `q(T_w)` continuous and C¹; `dq/dT_w < 0` on the transition branch | a kink here guarantees solver failure at traverse |
| 5.4 | **Film-branch `dt` limit** | 2.2 re-derived with a vapour near-wall cell | the acoustic constraint tightens by ~`c_l/c_v` |
| 5.5 | **Full boiling-curve traverse** | Tatsumoto, nucleate → DNB → film | the final case |

### 5.1 exposes a genuine model gap

Already measured: Zuber gives 97 kW/m² (+52%), `BubbleCrowding` gives 36 (−43%),
against a measured 64. They bracket it and neither is usable alone. Both are
*pool* boiling criteria being asked to predict a *flow* boiling CHF.

**Recommendation: add Katto–Ohno (or Shah) as an `AbstractCriticalHeatFlux`
implementation.** It is a small addition against the existing type hierarchy, and
it is the difference between a predictive model and one that has to be told the
answer through `FixedCriticalHeatFlux`. Without it the film-boiling result is not
a prediction.

### 5.3 is a stability gate, not a cosmetic one

The wall traverses the transition region during the run. If `q(T_w)` has a
discontinuity or a slope-sign error there, the wall-temperature bisection will
chatter between branches and the case will fail in a way that looks like a solver
problem. Sweep it and plot it before running any CFD through it.

---

## Ordering — the first four things to do

Ranked by information gained per unit of effort, not by stage number:

1. **0.1 — the linear-solver gate.** Everything else is unfalsifiable without it.
2. **0.4 — variable-`rho` well-balancedness.** A 1D column that either confirms
   or kills the leading unfalsified suspect in the existing notes.
3. **2.5 — compressible single-phase duct.** Separates conditioning from
   formulation; the split that currently cannot be made.
4. **4.2 — 0D boiling curve.** Seconds of work; tells you whether the closures
   are in the right decade for hydrogen before any solver effort is spent.

Then 0.6/0.7 (the Dittus–Boelter branch, already identified as the highest-value
first validation and still open), then Stage 1 in order.

---

## What was missing from the plan as stated

| # | Addition | Why |
|---|---|---|
| 1 | **Stage 0 in its entirety** | single-phase verification on the actual mesh, grid convergence, and a linear-solver gate. Without these, no later agreement with data can be attributed |
| 2 | **Variable-`rho` well-balancedness (0.4)** | the leading unfalsified suspect, and untested |
| 3 | **Stefan problem / sucking interface (3.2, 3.3)** | the standard analytic phase-change verification; entirely absent |
| 4 | **Kynch batch sedimentation (1.4)** | the analytic test for the coupled drift-flux system |
| 5 | **Bartolomei subcooled boiling, water (4.4)** | splits implementation error from cryogenic coefficient error — currently indistinguishable |
| 6 | **Adiabatic two-phase pressure drop (1.6)** | wall shear sets `h_c` in the RPI convective leg |
| 7 | **Condensation direction throughout** | only evaporation has ever been measured |
| 8 | **Acoustic CFL as a calibrated `dt` rule (2.2)** | turns a known instability into a design constraint |
| 9 | **Katto–Ohno CHF** | model gap; without it DNB is prescribed, not predicted |
| 10 | **Wall-boiling diagnostics in the output** | prerequisite for *any* boiling-curve comparison |
| 11 | **Table derivative consistency (2.1)** | `psi` vs `drho/dp` mismatch is an irreducible mass error |
| 12 | **Method of manufactured solutions** (optional) | exact answers for the alpha and energy equations at arbitrary complexity, if the rungs above leave ambiguity |

---

## Recording results

Each rung gets a row in a results table in this file, carrying: date, case file,
mesh, `dt`, steps run, physical time reached, the physics acceptance result, and
the four gates. A rung is **closed** only when the physics check and all
applicable gates pass. Anything else is recorded with the number of steps it
survived, never as "stable".

| Rung | Date | Case | Mesh | Physics | G1 | G3 | Status |
|---|---|---|---|---|---|---|---|
| 0.1 | 2026-08-19 | `src/Solve/Solve_2_monitor.jl` | — | instrument built + validated | — | — | **closed** |
| 0.4 A | 2026-08-19 | `test/0_TEST_CASES/0_4_hydrostatic_variable_density.jl` | quad 40² | `max\|U\| = 0.0` exactly | pass | pass | **closed** |
| 0.4 B | 2026-08-19 | same | quad 40² | `max\|U\| = 8.1e-12` | pass | pass | **closed** |
| 0.4 C | 2026-08-19 | same | quad 10²/40²/100² | imbalance at round-off | pass | pass | **closed** |
| 0.4 ctrl | 2026-08-19 | same | quad 10²/40²/100² | O(1), non-convergent, as required | — | — | **closed** |
| 0.4 D | 2026-08-19 | same | trig 218/3484/21098 | imbalance ~1st order | pass | pass | **closed** |
| 0.4 E | 2026-08-19 | same | quad 40² | seeded at equilibrium, `max\|U\| = 3.3e-11` | pass | pass | **closed** |
| 0.4 pipe | 2026-08-19 | `lh2_pipe_sector` | O-grid 25092 | axial imbalance 1.6e-7 | — | — | **closed** |
| 2.5 | 2026-08-19 | `test/0_TEST_CASES/2_5_compressible_duct_throughflow.jl` | quad 40² | both branches −0.06% of exact | pass | pass (0.039% over 32x dt) | **closed — FIXED** |
| 2.7 | 2026-08-19 | `test/0_TEST_CASES/2_7_table_derivative_consistency.jl` | — | tables 2nd-order consistent on-branch | — | — | **closed** |
| 2.6 | 2026-08-19 | `test/0_TEST_CASES/2_6_2_8_real_eos_duct.jl` | quad 40² | PengRobinson, −0.07% of exact | pass | pass | **closed** |
| 2.8 | 2026-08-19 | same | quad 40² | RealFluid, −0.06% of exact | pass | pass (0.047% over 16x dt) | **closed** |
| 2.9 | 2026-08-19 | `test/0_TEST_CASES/2_9_heated_duct_no_phase_change.jl` | quad 40² | Nu +0.31%, dT/dx −0.64% | pass | pass (0.000% over 16x dt) | **closed** |
| 3.1 | 2026-08-19 | `test/0_TEST_CASES/3_1_zero_d_relaxation.jl` | quad 10² | rate **+0.46%**; 1 defect (~4% budget) | pass | — | **open — 1 defect** |
| 4.2 / 5.2 / 5.3 | 2026-08-19 | `test/0_TEST_CASES/4_2_boiling_curve_closures.jl` | none (0D) | partition exact; film blend continuous; nucleate **0.79–1.07 after recalibration** | — | — | **closed** |

All 15 assertions pass. Regression unchanged at 3389.

---

## Findings so far

### 0.1 — the G1 instrument (built)

`monitor_linear_solves!()` / `linear_solve_report()` / `check_linear_convergence()`
in [`src/Solve/Solve_2_monitor.jl`](src/Solve/Solve_2_monitor.jl), hooked into
both `solve_system!` methods (Krylov and AMG). It recomputes the **true**
`||b - A x||` from the returned solution rather than trusting the method's
recurrence estimate, and reports per field: solves, at-itmax, unconverged, max
iterations, worst relative residual, worst residual/tolerance.

Two things it taught immediately, both about how to *ask* for a tolerance:

1. `rtol = 0` with a very small `atol` can demand a relative residual below the
   double-precision floor. Every solve then "fails" and the report is about the
   request, not the solver. The report now names this case explicitly.
2. **An absolute tolerance cannot serve a field whose right-hand side spans
   orders of magnitude over a run.** In a quiescent hydrostatic column the
   momentum RHS is ~1e-11 — there is nothing to solve — while during startup it
   is ~5e2. A fixed `atol` below the quiescent floor reported 999 of 2500 solves
   as failures on a case whose answer was correct to 4e-14.

`check_linear_convergence` therefore takes a `slack` (default 10): a Krylov
recurrence residual can differ from the true residual by a small factor — 2.3x
was measured here on a healthy solve — while the failure this gate exists to
catch was **three to four orders** short.

### 0.4 — CLOSED. The leading suspect is eliminated.

**The imbalance is algebraic and needs no time stepping.** Equilibrium exists iff
some cell field `p_rgh` satisfies, on every interior face,

    p_rgh[c2] - p_rgh[c1] = -ghf[f]*(rho[c2] - rho[c1])

so the irreducible imbalance is `min over p_rgh of ||snGrad(p_rgh) + ghf*snGrad(rho)||`
— a least-squares problem, solved exactly in seconds. Reported below as a fraction
of the buoyancy term's own norm, i.e. *the fraction of buoyancy that cannot be
balanced*.

| configuration | coarse | medium | fine | order |
|---|---|---|---|---|
| **smooth rho, gravity-aligned quad** | 2.53e-15 | 1.73e-13 | 2.22e-12 | round-off |
| **positive control** (g tilted 45°) | 1.85e-1 | 1.80e-1 | 1.79e-1 | **O(1)** |
| **non-orthogonal (triangular)** | 6.71e-4 | 8.40e-5 | 3.20e-5 | **~1.1–1.5** |

**On a gravity-aligned mesh the scheme is EXACTLY well balanced** — for uniform,
piecewise-constant *and* smooth `rho`. The residual is round-off, growing only with
problem size as a Poisson solve's conditioning does. **The hypothesis is dead.**

The **positive control validates the instrument**: tilting gravity so `g × ∇rho ≠ 0`
creates a genuine baroclinic torque that no pressure field can cancel, and the
measure correctly reports O(1) *and correctly refuses to refine it away*. Without
that control, "everything is tiny" would prove nothing.

**On a non-orthogonal mesh there IS a real imbalance**, converging at only ~1st
order rather than 2nd. Present, bounded, refinable — but 1st order.

**Dynamic confirmation.** Seeding `p_rgh` with the least-squares equilibrium
instead of zero drops the settled `max|U|` on the 40² mesh from 3.0e-8 to
**3.3e-11** — a factor of 900 — and it is now *flat in time* (3.35e-11 at 0.08 s,
3.28e-11 at 0.32 s) rather than ringing. That is direct confirmation that
everything previously measured was the internal gravity wave excited by starting
away from equilibrium, not an imbalance.

### The real pipe O-grid, measured directly

The same instrument, run on `examples/0_GRIDS/lh2_pipe_sector` (25 092 cells,
gravity along the pipe axis):

| density gradient | fraction |
|---|---|
| **axial** (`∇rho ∥ g`, what uniform wall heating produces) | **1.60e-7** |
| radial (`∇rho ⊥ g`) | 1.30e-4 |

The O-grid's non-orthogonality lives entirely in the cross-plane, where an axial
stratification has `Δrho = 0`, so it never enters. **The buoyancy discretisation is
not the pipe case's problem.**

The radial figure is *not* a defect measurement: with `∇rho ⊥ g` there is no
equilibrium even in the continuum, so that residual mixes real physics (natural
convection) with numerics and the measure cannot separate them.

### RETRACTED — the earlier "findings" from this rung

An earlier version reported a settled residual *growing* under refinement at order
**-1.9**, and growth in time on a non-orthogonal mesh. Both were read from runs of
0.32 s, and both are wrong.

The column is stably stratified, so it supports internal gravity waves;
`p_rgh = 0` rings them; viscous damping of the domain mode is `nu*(pi/L)^2 ~ 1.6e-4
1/s`, a decay time of ~6000 s. Sampling `max|U|` on the 40² mesh out to 20 s shows
it decay by 6.5x, **turn around at t ~ 14 s and climb again** — a ~27 s oscillation
(`omega ~ 0.24 N`, with `N = 0.967` rad/s giving a 6.50 s Brunt–Vaisala period).
Every reading at 0.32 s was one arbitrary phase of it, and the 100² sequence
1.68e-7 → 1.22e-7 → **2.22e-8** → **1.01e-7** at 0.32/0.64/1.28/2.56 s is the same
ringing sampled at four phases.

Worth recording as a method failure, not just a wrong number: this document opens
by warning against concluding from short runs, and the first measurement it took
did exactly that.

### Open items from 0.4

- **The ~1st-order convergence on non-orthogonal meshes.** Real but not urgent:
  the pipe O-grid does not exercise it for an axial stratification. It would
  matter for a case with a genuinely mesh-oblique density gradient.
- `T` drifts 1.94e-5 K on **every** mesh regardless of velocity, which is not
  advective and not currently explained. Negligible against a 30 K stratification
  (7e-7 relative), so it does not affect the result, but it is unaccounted for.

### 2.5 — CLOSED. The compressible formulation is defective, not the O-grid.

Plane channel, orthogonal cells at aspect ratio 1, laminar, adiabatic, no gravity,
single phase, **started from the exact Poiseuille solution** (parabolic profile +
linear pressure drop). Known answer `dp/dx = 12*mu*U_b/H^2`. G1 enforced.

| arm | dp/dx (exact 0.27875) | outcome |
|---|---|---|
| **A** incompressible control (`ConstEos`) | 0.27695 (−0.64%) | **stable** |
| **B** compressible (`IdealGas`), all else identical | — | **DIVERGED** |
| **D** `pressure_work_relax = 0` | 0.26712 (−4.17%) | stable |
| **D** `expansion_relax = 0` | 0.26719 (−4.15%) | stable |
| **D** both `= 0` | 0.26712 (−4.17%) | stable |

**The formulation is defective in through-flow, independently of the O-grid.**
Option (ii) — conditioning — is eliminated: this is 1600 orthogonal unit-aspect
cells, and it fails in one second.

**Two corrections to `dev_notes_LH2_pipe_boiling.md`:**

1. **It is NOT an acoustic CFL limit.** That reading predicts small `dt` is safe.
   Measured, the scaling is *inverted*:

   | dt | ×acoustic limit | outcome |
   |---|---|---|
   | 1e-5 | 0.1× | **DIVERGED** |
   | 5e-5 | 0.7× | survives, dp/dx **+12844%** |
   | 2e-4 | 2.8× | stable, +3.97% |
   | 1e-3 | 13.9× | stable, −0.88% |

   Smaller `dt` is *worse*. Whatever the mechanism is, its growth per unit
   physical time increases as `dt` falls — the signature of a term that does not
   scale correctly with `dt`, not of a wave running past its Courant number. The
   pipe's `dt = 2e-6` being "25x over the limit" was therefore the wrong
   diagnosis, and cutting `dt` to 1e-8 helping was a coincidence of that case.

2. **"Neither leg alone suffices" was concluded from DAMPING, not disabling.**
   The recorded sweep used `pressure_work_relax` = 0.5 and 0.05, never 0. Set
   either `pressure_work_relax` or `expansion_relax` to exactly **0** and the case
   is stable on its own. Cutting *either* arrow opens the loop

       dT -> expansion -> dp -> dp/dt -> S_T = beta*T*dp/dt -> dT

   which confirms the closed-loop reading and makes it actionable: only one of the
   two terms needs a correct implicit or lagged treatment, not both.

   Damping is not a substitute for opening it — `pressure_work_relax = 0.05` still
   gives `dp/dx = 0.081` against an exact 0.279, i.e. bounded but 70% wrong.

**Secondary finding.** Even with the loop opened the compressible branch returns
`dp/dx` 4.2% low, against 0.6% for the incompressible control on the same mesh —
20x less accurate on a quantity with an exact answer. Bounded, but not right, and
it should be understood before the branch is trusted quantitatively.

### 2.5 — THE FIX

**Diagnosis.** The explicit loop was the solver recovering the **isentropic**
compressibility by iterating on the **isothermal** one. Substituting the
pressure-work temperature response back into the expansion source,

    expansion = beta*(dT/dt)_other + (beta^2*T/(rho*cp))*dp/dt

and moving the second part onto the left-hand side leaves the time coefficient

    psi_s = psi_T - beta^2*T/(rho*cp)

which is the exact thermodynamic identity between the two compressibilities. For
an ideal gas `psi_T = 1/p` becomes `psi_s = 1/(gamma*p) = 1/(rho*c^2)` — the
coefficient that carries the acoustic wave speed, which is what the pressure
equation needed all along.

So this is not damping and not a numerical fudge: it is supplying a
thermodynamic coefficient directly instead of iterating towards it.

**Implemented** as `Fluid{Multiphase}(..., thermo_acoustic = :implicit)`, the new
**default**; `:explicit` retains the old behaviour. Three pieces:

1. `_update_psi_isentropic!` — the corrected time coefficient. The two `betaT`
   factors are *not* the same quantity and must not be merged: the expansion
   source carries the weights of the chosen pressure form, while the energy
   equation's pressure work is always volume-weighted. Merging them is correct
   only for `pressure_form = :volume` and wrong by a factor of `rho` otherwise.
2. `_update_expansion_implicit!` — subtracts `S_T/rho_cp`, the increment the
   pressure-work source actually produced. Subtracting the *source the energy
   equation was given*, rather than a modelled estimate of its effect, is what
   makes the two halves cancel to the solver's own discretisation.
3. A floor at `0.1*kappa_T`. `kappa_s/kappa_T = 1/gamma`, so the correction can
   never legitimately exceed ~40%; the floor only bites on inconsistent property
   tables and keeps the time coefficient positive.

**Measured on rung 2.5:**

| arm | dp/dx (exact 0.27875) | outcome |
|---|---|---|
| incompressible control | 0.27858 (**−0.06%**) | stable |
| compressible, **implicit** (new default) | 0.27858 (**−0.06%**) | stable |
| compressible, `:explicit` (legacy) | — | **DIVERGED** |

**G3 restored.** Across a 32x range of `dt`, spanning 0.14x to 4.4x the acoustic
limit, `dp/dx` moves by **0.039%** — and the two smallest steps are exactly where
the legacy path diverged.

For comparison, the workaround of *disabling* a leg (`pressure_work_relax = 0`)
gave −4.17%. The implicit coupling is **50x more accurate** than switching the
term off, because it corrects the physics rather than removing it.

**No regression, and a large improvement on the validated case.** Full suite
unchanged at 3389 passing. On the sealed-tank acceptance test
(`unit_test_compressible_ullage.jl`, `dp/dt = R*Q/(V*cv)` exactly):

| pressure form | error before | error after |
|---|---|---|
| `:mass` | −1.02% | **−0.013%** |
| `:volume` | −1.01% | +1.27% |

The mass form improves by ~80x. That asymmetry is expected and is itself a check
on the derivation: `psi = drho_m/dp` is a true derivative in the mass form, so the
correction is exact there, and only approximate under the volume form's per-phase
weighting. The notes already identify `:mass` as the consistent choice for a
`Mixture`; this is independent evidence for it.

### `pressure_form` — `:mass` stays OPT-IN

Attempted as the default and **reverted**. `:mass` scales the Laplacian by
`rho_f` and the resulting pressure matrix is **not symmetric**, so `Cg()` is
invalid for it — and `Cg()` is what the existing multiphase cases use for
`p_rgh`. Switching the default broke `2d_multiphase_gravity` and
`2d_multiphase_hydrostatic` immediately.

Promoting it therefore means changing every case's pressure solver to
`Bicgstab()` at the same time, which is a separate decision. The accuracy case
for `:mass` is unchanged and recorded above (−0.013% vs +1.27% on the sealed
tank; 1670x less mass drift).

### 2.6 / 2.7 / 2.8 — real equations of state, CLOSED

**2.7 first, because it gates 2.8.** `RealFluid` tabulates rho, psi and beta as
three independent tables; the isentropic correction mixes all three, so their
mutual consistency is now load-bearing rather than cosmetic. Checked by central
difference on the same grid, **on-branch only**:

| branch | rms psi (41→161) | rms beta (41→161) | order |
|---|---|---|---|
| liquid | 7.1e-5 → 3.7e-6 | 6.4e-4 → 4.0e-5 | ~2 |
| vapour | 1.5e-3 → 9.0e-5 | 5.4e-3 → 3.2e-4 | ~2 |

They are consistent, and converge. Peng-Robinson, as the control, agrees with its
own `d(rho)/dp` to 2.7e-7 (liquid) and 4.5e-6 (vapour) — one cubic, so consistent
by construction.

**On-branch only, and why it matters.** An earlier version sampled the whole
(p,T) box and reported relative errors of **1e15**. That was the metric failing,
not the tables: off-branch, `build_property_tables` continues metastably and then
falls back to the saturation line, so `rho` is pinned while `psi` and `beta` are
not — they are not derivatives of a common `rho` there and cannot be.

**This changed the fix.** The isentropic correction as a fraction of `psi`:

| | on branch | off branch |
|---|---|---|
| liquid | 0.59 (gamma = 2.4) | **9.6** |
| vapour | **0.907** (gamma = 10.7, at 1.01 MPa / 32.5 K) | **11.2** |

The vapour figure is near-critical, where `cp` genuinely diverges and a large
gamma is **physical** — and it sat right against the `0.1*kappa_T` floor I had
originally written, which would have clipped a correct state. The floor is now
**0.01** (gamma up to 100). It is not a safety margin: off-branch the correction
reaches ~10x `psi`, and without a floor `psi` would go negative in every cell
that contains none of that phase.

**2.6 and 2.8 on the duct.** `dp/dx = 12*mu*U_b/H^2` contains no density, so every
arm must return the same number and any spread is the EoS path alone:

| EoS | dp/dx (exact 16.800) | |
|---|---|---|
| `ConstEos` (control) | 16.7901 (−0.06%) | G1 ok |
| `PengRobinson` | 16.7882 (−0.07%) | G1 ok |
| `RealFluid` | 16.7907 (−0.06%) | G1 ok |

Spread across the three: **0.015%**. G3 on the tabulated arm over a 16x `dt`
range: **0.047%**.

**Peng-Robinson's density error is printed, not hidden**: 79.807 against the
Helmholtz 66.489 at 0.4 MPa / 24 K, **+20.0%**. Consistent with its own header
(H2 has a negative acentric factor, outside the alpha-function's fitted range).
It is carried as a *numerical* stepping stone — smooth, analytic, internally
consistent — not as a validation EoS. Note it also supplies **no saturation
curve**, so any use past Stage 2 needs one paired in.

### 2.9 — heated duct, no phase change. CLOSED.

**The first rung where the fix is genuinely exercised.** Rungs 2.5-2.8 were all
adiabatic: `dT/dt` was order 1e-6 K, so the `expansion` source was doing almost
nothing and the isentropic correction passed a test that barely loaded it. Here
the wall flux drives it properly.

Two exact answers for laminar plane-channel flow, uniform `q_w` on both walls:

| quantity | exact | control (`ConstEos`) | compressible (`IdealGas`) |
|---|---|---|---|
| `dT/dx = 2*q_w/(rho*cp*U_b*H)` | 0.49347 K/m | 0.48926 (−0.85%) | 0.49031 (−0.64%) |
| `Nu` on `D_h = 2H` | 8.2353 | 8.2549 (**+0.24%**) | 8.2605 (**+0.31%**) |

G1 passes on both. **G3 is exact to the digits printed**: over a 16x `dt` range the
Nusselt number moves by 1.8e-5, i.e. **0.000%**.

`dT/dx` is energy conservation and needs no development. `Nu` does, and the
laminar thermal development length here far exceeds the channel, so the
temperature is **seeded developed and imposed at the inlet** through a
`DirichletFunction` — the same device that closed rungs 0.4 and 2.5. The profile
was derived rather than looked up:

    theta(y) = (q_w/k)*(2*y^3/H^2 - y^4/H^3 - y) + C

with `C` fixed by a vanishing bulk mean, giving `theta(0) = 0.242857*q_w*H/k` and
`Nu = 2/0.242857 = 8.2353` — which recovers the tabulated value, confirming the
seed is the right profile and not merely a plausible one.

**Control arm.** With `q_w = 0` the residual axial gradient is 1.3e-5 K/m, which
is **2.7e-5 of the heated gradient** on a 300 K field. Judged as a fraction of the
signal rather than against an absolute number — what matters is that it cannot
contaminate the measurement.

(An early version of that arm still seeded the *heated* profile while setting
`q_w = 0`, so it advected the gradient it was handed and reported 0.474 K/m as if
it were physics. Worth recording: a control that shares its initial condition
with the case it controls is not a control.)

**What this unblocks.** Every phase-change model is driven by `T - T_sat`, so this
is the field Stages 3 and 4 will all be reading. It is now verified to ~0.3% on a
quantity with an exact answer.

### 3.1 — 0D relaxation. Rate law CORRECT, three defects found.

Uniform sealed box, no flow, no gradients, no wall boiling. Saturated LH2 at
0.4 MPa (`T_sat = 26.077 K`, `h_fg = 393.5 kJ/kg`), Lee model with
`DispersedBubbles`, tabulated property values. Exact answer:

    T - T_sat = (T_0 - T_sat)*exp(-t/tau),   tau = rho_cp/(C*h_fg)
    C = beta*alpha_l*rho_l/T_sat * 6*alpha_l*(1-alpha_l)/d

**PASSES — the rate law.** Measured `tau = 4.020 s` against an analytic
`4.437 s`, **−9.4%**. That single number validates the Lee rate, the kinetic
prefactor `sqrt(1/(2 pi R_sp T_sat))`, the `DispersedBubbles` area closure and
the latent sink together. Condensation relaxes in the right direction, ~117x
slower, which is the model's own weighting asymmetry (`(1-alpha)*rho_v = 0.484`
against `alpha*rho_l = 56.7`) and not an error.

**A setup trap worth recording.** The first run had the superheat *growing*
(fitted `tau = -27.8 s`). Not a code fault: evaporation in a SEALED box creates
volume, the box pressurises, and `S_T = beta*T*dp/dt` heats the fluid. At 0.2 K
the pressure work won; at 5 K the latent sink dominated and it relaxed but made
115% of the vapour the sensible heat could pay for. Both are real physics of a
sealed box and neither is the rate law, so `pressure_work_relax = 0` isolates it
out — rungs 2.5 and 2.9 already verify that path.

#### DEFECT 1 — the tracked/liquid convention. **FIXED.**

`phase_change_rate!` passed the **tracked** `alpha` into the Lee weight, while the
model documents the **liquid** fraction and `multiphase_liquid_phase` states that
"the phase-change sink, its sign, ..." must use `liquid_phase`. The solver already
computed `alpha_liq` for exactly this and gave it to **wall boiling** — but not to
the bulk rate.

Measured on arm C (identical physical state, phases swapped):

| | tau |
|---|---|
| before | 25.0 s against 4.02 s — **factor of 6.2** |
| after | **+0.80%** |

Fixed by filling and passing `alpha_liq`. Arm C is retained as a regression guard.

#### The Lee API — prescribed coefficient, not a derived one

Changed at the same time, and the two interact. `Lee` previously took an
accommodation coefficient `sigma` from which the relaxation parameter was
**derived**,

    beta = sigma*sqrt(1/(2*pi*R_sp*T_sat))*L*rho_l/(rho_l - rho_v)

and the result was then **multiplied by the interfacial area density** `a_i`. So
`sigma` was neither the `r` of the published Lee model nor in its units, and the
effective coefficient carried a hidden `alpha*(1 - alpha)` factor.

Now:

    Lee(r = 100.0)        # relaxation coefficient [1/s], both branches
    mdot = r*alpha_l*rho_l*(T - T_sat)/T_sat        [kg/m^3/s]

with a new trait `uses_interfacial_area(::Lee) = false`, because `r` is already
volumetric. `Schrage` and `ModifiedEnergyJump` genuinely return per-area fluxes
and keep the `a_i` scaling. `Lee(sigma = ...)` and `Lee(R = ...)` now **throw**
with an explanatory message rather than being silently reinterpreted — the two
differ by orders of magnitude and a quiet change would corrupt every existing case.

**Effect on the rate check:** arm A went from **−9.4%** to **+0.46%** against the
analytic tau. Removing the spurious area factor is most of that.

#### DEFECT 2 — **LARGELY RETRACTED.** Real residual is ~4%, not +77%.

The clamp hypothesis was tested and is **dead**, and testing it exposed the
measurement error underneath.

**The clamp never fires.** In the 0D box `alpha` runs 0.900 -> 0.892 and never
approaches a bound, so `clamp(a, 0, 1)` in `_apply_phase_change_alpha!` is never
active. It cannot explain anything here.

**The +77% was a zero baseline.** `vap_mass` was sampled before the first `run!`,
when `update_phase_state!` had not yet executed and the vapour density field was
still all zeros. The difference therefore counted the vapour that had been present
all along as newly created. Taking the baseline after one step:

| | before | after |
|---|---|---|
| dm_l | — | −1.1552e-3 kg |
| dm_v | +4.89e-3 kg | +1.2089e-3 kg |
| imbalance at 0.2 K | **+76.5%** | **+6.16%** |

**What survives is real but modest.** Sweeping the superheat squeezes out the
linearisation in `expected` (which holds `rho_cp` at the INITIAL alpha):

| dT0 | 0.02 K | 0.05 K | 0.2 K | 1.0 K | 5.0 K |
|---|---|---|---|---|---|
| error | **+4.12%** | +4.48% | +6.16% | +12.40% | +23.77% |

It does **not** vanish as `dT0 -> 0`, so ~4% is genuine. Everything above that is
`expected` crediting more sensible heat than the shrinking liquid actually holds.

Total mass drift over 14 s at 0.2 K is 5.8e-4 relative - small, but growing.

#### DEFECT 3 — **RETRACTED.** Condensation does consume vapour.

The reported "+3.63e-3 kg created while subcooled" was the same zero-baseline
error. With the baseline taken after one step the vapour change is
**−1.218e-4 kg**: negative, as it must be. Arm B now passes as a real assertion.

#### A separate, unexplained drift in the VOF unit test

`unit_test_phase_change.jl` still shows Lee at **2.7e-2** mass drift against
Schrage 9.2e-5 and MeJ 5.3e-8. That case is NOT the 0D box: it is VOF with
`setField_Box!`, so it HAS pure cells and a sharp interface, and there the clamp
CAN fire - Lee is no longer area-scaled, so it evaporates in pure liquid cells
where `a_i = 0` would previously have silenced it. Marked `@test_broken` and left
for a rung that isolates it; the 0D result does not transfer.

#### Method note

Two of the three "defects" reported from this rung were errors in the
instrument, not the code, and both came from the same habit: differencing against
a baseline sampled before the solver had populated its own fields. The rate check
was unaffected because it reads `T`, which IS initialised. Worth remembering that
a derived quantity is only as trustworthy as the first sample it is measured from.

### VOF alpha clamp — deferred deliberately

Commented in place at `_apply_phase_change_alpha!` and left. It affects the VOF /
sharp-interface path only: Lee drifts 2.7e-2 against Schrage 9.2e-5 and MeJ 5.3e-8
in `unit_test_phase_change.jl`, because Lee is no longer area-scaled and so
evaporates in pure cells where `a_i = 0` used to silence it, and the `clamp(a, 0,
1)` then discards the excess. `Mixture` does not exercise it — rung 3.1 shows
`alpha` running 0.900 -> 0.892, never near a bound. Standing `@test_broken`.

### 4.2 / 5.2 / 5.3 — the boiling curve from the closures, no CFD

`wall_heat_partition` is a pure function of a `BoilingState`, so the whole curve
costs milliseconds. Compared against
`data/tatsumoto2014_D6_L250_0.4MPa_5.53ms.csv` — the digitised measurement at
exactly this operating point. **(The plan's earlier "experimental data not in the
repo" note was wrong; it has been there all along.)**

`h_c` is Dittus-Boelter at the experiment's Reynolds number (Re_D = 2.36e5,
Nu = 556, h_c = 9.35e3 W/m^2/K) — stated rather than tuned, as the stand-in for
what the wall function will produce in CFD.

**PASSES:** the partition is exact and complete, every component non-negative,
`A_b` bounded, and `N_a = q_e = 0` at zero superheat.

**THE FINDING — the shape is wrong, not the constant.**

| dT_sup | q_RPI | q_measured | ratio | q_e share |
|---|---|---|---|---|
| 0.70 | 6.75e3 | 4.66e3 | 1.451 | 3.7% |
| 1.10 | 1.07e4 | 1.06e4 | 1.010 | 5.3% |
| 1.67 | 1.63e4 | 5.88e4 | **0.278** | 7.3% |

The measured branch is near-VERTICAL — q runs 4.3 -> 64 kW/m^2 while `dT_sup`
moves only 0.66 -> 1.67 K. RPI here is dominated by single-phase CONVECTION
(q_e is under 10% throughout), so `q(dT_sup)` comes out nearly LINEAR and falls
progressively behind.

The cause is `LemmertChawla`'s exponent: `N_a ~ (m*dT_sup)^n` with **n = 1.805**
is a water fit, while this repo's own film-boiling notes cite **n ~ 21.17 for
LH2**. An exponent an order of magnitude too low cannot produce a vertical branch
at ANY prefactor `m`, so this is not removable by rescaling — it needs the
cryogenic fit `calibrate_rpi_lh2.jl` exists to produce. Pinned as `@test_broken`
so a recalibration has a baseline.

**Film boiling (5.3) PASSES.** With `FixedCriticalHeatFlux(q = 64 kW/m^2)` (the
measured value), `Berenson` minimum film and `Bromley` HTC, the blended curve:

- peaks at exactly **6.40e4 W/m^2**, i.e. the CHF — the nucleate cap is working;
- turns over at `dT_sup = 9.2 K`, which is `dT_lo` by construction;
- is continuous: largest step **0.77% of peak** over a 0.05 K sweep to 80 K;
- shows no trace of the 10^7 kW/m^2 mid-transition spike the cap exists to prevent.

A measurement note: continuity must be judged against the PEAK flux, not the local
value. Normalising locally reported a "101% step" at `dT_sup = 0.1 -> 0.2 K`, which
is just `q_c = h_c*dT` doubling near zero — perfectly smooth.

### RPI recalibrated for LH2 — the shape problem is fixed

Run `test/0_TEST_CASES/calibrate_rpi_lh2.jl`. No CFD; the partition is algebraic,
so the whole parameter space sweeps in seconds.

**Two corrections to the calibration itself, both material.**

1. **The old grid was choosing the answer.** The first run returned `n = 14.000`
   — exactly the top of `N_GRID = range(1.0, 14.0)`. An optimum on a boundary is
   not an optimum. Extended to `n ∈ [1, 40]` (and `m` down to 1e-2, since
   `N_a = (m*dT)^n` needs a smaller `m` as `n` grows); the optimum is now interior.

2. **`h_c` must come from the data, not from Petukhov.** The lowest-flux points of
   a boiling curve ARE single-phase convection — the wall is barely superheated
   and nucleation does nothing — so `q = h_c*dT_sup` determines `h_c` directly.
   It is also the one quantity NOT degenerate with `(m, n)`, being pinned exactly
   where they are inactive.

   | | h_c [W/m^2/K] |
   |---|---|
   | Petukhov at y+ = 40 | 14199 |
   | **from the data's low-flux limit** | **7045** |

   The correlation is **2x high**. The pipe notes record the same gap from the
   other side (measured h_conv 3692-5388 against Dittus-Boelter ~10,200). Fitting
   the site density against a 2x-high `h_c` just makes nucleation absorb the error.

**Fitted result:**

    site_density       = LemmertChawla(m = 3.0, n = 7.798)
    departure_diameter = TolubinskyKostanchuk(d_ref = 0.00111, d_max = 0.004441)

`d_ref` is the Fritz diameter for LH2 (1.11 mm), not the 0.6 mm water value.

| | RMS(log dT) |
|---|---|
| shipped water fit | 0.8875 |
| **LH2 fit** | **0.0673** |

A factor of **13**. The ridge is tight — 11 of 12100 grid points within 10% of
best, `m` spanning 1.77-4.12 and `n` 6.37-11.73 — so `(m, n)` are far better
determined than they were, though still traded off against each other and not to
be quoted as physical.

**Note on `n`.** This repo's film-boiling comments cite `n ~ 21.17` for LH2; the
fit here gives **7.8**. Different, and not reconciled — the 21.17 presumably came
from a different `h_c`, `d_ref` or dataset. The value here is the one consistent
with this curve and this `d_ref`.

**Effect on rung 4.2 — the shape is fixed:**

| | water fit | LH2 fit |
|---|---|---|
| ratio q_RPI/q_measured across the branch | 0.278 – 1.451 | **0.794 – 1.069** |
| q_e share at top of branch (dT = 1.67 K) | 11.9% | **87.3%** |

The water fit had single-phase convection carrying 90% of the flux, so the curve
came out linear where the measurement is near-vertical. The recalibrated closure
has evaporation carrying 87%, which is what nucleate boiling looks like, and the
near-vertical branch appears: `q` runs 4.97 → 46.7 kW/m^2 while `dT_sup` moves
only 0.70 → 1.67 K.

Film boiling (5.3) is unaffected and still passes: peak exactly at CHF, turnover
at `dT_lo = 9.2 K`, largest step 0.77% of peak.

**Caveats carried from the script:** calibrated at ONE pressure and ONE velocity
(0.4 MPa, 5.53 m/s); `m` is an amplitude that has absorbed everything degenerate
with it; valid only up to CHF. And `h_c = 7045` is from the experiment — if the
CFD's wall function produces something materially different, the convection/
evaporation split moves and this fit moves with it.

### Open items from 2.5

- **`:volume` form** now carries +1.27% on the sealed tank against the mass form's
  −0.013%. Not a regression (it was −1.01%), but it makes `:mass` the clearly
  preferable default and that decision should be taken deliberately.
- Not yet run with tabulated `RealFluid` properties instead of `IdealGas`, or with
  turbulence. `IdealGas` was chosen so a failure would be the formulation and not
  the tables — that worked, and the tables are now the *next* variable to add, not
  a confound to eliminate.

### 0.1 addendum — the G1 criterion, third revision

The 10² G1 failure is resolved, and its resolution is the general one. All 8
failures (of 1920) were `U_y`, all at `iterations = 0`, with `|b|` between 2.1e-6
and 3.8e-5 against `atol = 1e-6`, while the peak `|b|` for `U_y` over the run was
5.1e-2 — solves of a null system, at 7e-4 of the field's working scale.

Raising `atol` fixed this twice and it reappeared at a new scale on a new mesh.
The judgement "there is nothing to solve" cannot come from a user-supplied
absolute number when a field's RHS spans four orders over a run. `is_negligible`
now makes it against **the field's own peak `|b|`**, tracked automatically, with
the excused count reported rather than hidden (`n_negligible` in the report).

