# LH2 forced-convection pipe boiling — implementation notes

Companion to [`dev_notes_LH2_implementation_plan.md`](dev_notes_LH2_implementation_plan.md)
(the sealed-tank work this reuses) and
[`test/0_TEST_CASES/3d_LH2_pipe_forced_convection.jl`](test/0_TEST_CASES/3d_LH2_pipe_forced_convection.jl)
(target case).

Reference: Tatsumoto, Shirai, Shiotsu, Hata, Naruo, Kobayasi & Inatani,
*Forced convection heat transfer of saturated liquid hydrogen in
vertically-mounted heated pipes*, AIP Conf. Proc. **1573**, 44–51 (2014).

---

## Status

| Step | State |
|---|---|
| 1. Real-fluid property tables | **done** — `RealFluid`, validated against the direct EOS |
| 2. Variable cp/k/mu in the multiphase path | **done** — per-phase face fields |
| 3. Turbulent two-phase conductivity | **done** — replaced the laminar-only error |
| 4. RPI wall boiling + swappable sub-models | **done** — 95 unit tests |
| 5. Solver coupling (alpha / pressure / energy) | **done** |
| 6. 90° O-grid pipe mesh, y+ 30–50 | **done** — 28 mesh tests, y+ 33–48 |
| 7. Case runs to a usable solution | **BLOCKED** — see below |
| 8. Validation against the paper | pending — blocked by 7 |

---

## Blocker — the compressible pressure equation is sealed-domain only

**This is a pre-existing limitation of `Solvers_5_Multiphase.jl`, exposed rather
than introduced by this work.**

`solve_pressure_compressible!` holds the pressure-equation time-term reference at
the value from the *start* of the step. Its docstring explains why that is
necessary: inside the PISO corrector loop the reference would otherwise move with
every corrector, and the pressure rise would scale with `inner_loops`. It was
written for the K-Site tank, and the accompanying `pref` check refuses a
reference cell on the grounds that "the compressibility term already fixes the
pressure level".

Both of those statements are true for a **sealed** domain. Neither holds for a
flow-through pipe, where the pressure level is set by the outlet Dirichlet
condition — so the outlet BC and the compressibility term are fixing the same
quantity, and they fight.

### Evidence

Identical mesh, turbulence model, boundary conditions and inlet velocity
(5.33 m/s); 150 steps unless stated. The only variable is the equation of state.

| Configuration | max\|U\| | Outcome |
|---|---|---|
| `RealFluid` both phases, no phase change, q = 3e4 | 5.2e5 m/s | diverged |
| `RealFluid` both phases, Lee only, q = 3e4 | NaN | diverged |
| `RealFluid` both phases, RPI only, q = 3e4 | NaN | diverged |
| `RealFluid` both phases, no phase change, **q = 0**, 20 steps | 5.2e3 m/s | diverged |
| `ConstEos` both phases, no phase change, q = 0 | **5.34 m/s** | stable |
| `ConstEos` both phases, no phase change, q = 3e4 | **5.34 m/s** | stable |

The last two settle at 5.34 m/s against a 5.33 m/s inlet, with the temperature
uniform when q = 0 and rising sensibly when q = 3e4. So the through-flow boundary
conditions, the O-grid, the wall functions, the drift-flux `Mixture` path and the
boiling models are all exonerated: the failure is specific to the compressible
pressure path, and it occurs with **no phase change and no heating at all**.

### FOUND AND FIXED — `p_abs_prev` was seeded without the hydrostatic term

Located by an instrumented field-by-field diff of two runs differing *only* in
the vapour's `rho_model` (`ConstEos` vs `TabulatedEos`), with a constant-property
liquid and `alpha = 1`, so every blended property is provably identical.

After one step, `rho`, `rhof`, `nu`, `nuf`, `rho_cp` and `keff` were bit-identical
while `S_T` differed by `1.378e8` (the `ConstEos` case being exactly zero).

**Cause.** In `MULTIPHASE`:

```julia
p_abs_prev = ScalarField(mesh); initialise!(p_abs_prev, p_operating)   # uniform
```

but `absolute_pressure!` computes `p_abs = p_rgh + rho*gh + p_operating`. The
first step therefore saw

    dp/dt = (rho*gh + p_rgh_0)/dt = -172/2e-6 = -8.6e7 Pa/s

— the entire hydrostatic head divided by one time step, produced by nothing
physical. Both downstream quantities match theory exactly:

| quantity | predicted | measured |
|---|---|---|
| `S_T = beta*T*dp/dt` | 1.38e8 W/m^3 | **1.378e8** |
| `dT = beta*T*rho*gh/(rho*cp)` | 1.97e-4 K | **1.974066e-4** |

**Fix.** Seed `p_abs_prev` from the actual initial absolute pressure:

```julia
absolute_pressure!(p_abs, p_rgh, rho, gh, p_operating, config)
@. p_abs_prev.values = p_abs.values
```

After the fix, step 1 is **bit-identical across all 16 diffed fields**.

Note the temperature error `beta*T*rho*gh/(rho*cp)` is **independent of `dt`**, so
refining the time step never revealed it. It is pre-existing and affects the
K-Site path too, where it damps out in a large slow sealed tank. No regression:
3300 tests pass, including the compressible-ullage suite that validates K-Site's
dp/dt to 0.09%.

### MEASURED — the discrete vapour mass balance is violated by ~100%

The temperature form of the energy equation uses `S_T = -mdot*h_fg`. Expanding
the mixture enthalpy `rho_m h_m = rho_cp T + (1-alpha) rho_v h_fg` shows that
source is exact **only if** the discrete vapour mass balance holds:

    d((1-alpha) rho_v)/dt + div((1-alpha) rho_v u_v) = mdot

Measured directly (RPI on, q_w = 3e4, wall cells):

| step | max\|mdot\| | max\|residual\| | residual/mdot | energy error |
|---|---|---|---|---|
| 1 | 1847.6 | 1834.0 | 0.99 | 6.0e8 W/m^3 |
| 2 | 2076.2 | 2140.5 | 1.03 | 7.0e8 W/m^3 |
| 19 | 764.5 | 694.4 | 0.91 | 2.3e8 W/m^3 |

The balance is not approximately satisfied - it is violated essentially
completely, every step. Multiplied by `h_fg` = 327400 J/kg that is a spurious
energy term of **2-7e8 W/m^3 against a wall input of 8.8e8 W/m^3**, i.e. 23-80%
of the applied heat flux, injected exactly where the cooling is observed. This
is the first measured mechanism whose magnitude matches the symptom.

**Why the balance fails.** The alpha equation is advanced in ADVECTIVE form,

    alpha = alpha_prev - dt*(div(alpha*u) - alpha*div(u))

which is exact only when `div(u) = 0`. With boiling, `div(u)` reaches ~84 1/s in
the near-wall cells against a physical phase-change sink `Gamma/rho_l` of ~32
1/s - the rearrangement term is several times larger than the source beside it.
The compressibility term `-(alpha_i/rho_i) Drho_i/Dt` (STAR-CCM+ Eq. 2900) is
also absent, and the phase-change source is applied AFTER the MULES limiter.

**Attempted fix, REVERTED.** Switching to the conservative form on the
compressible path made it markedly worse:

|  | vapour mass residual | alpha_min | max\|U\| |
|---|---|---|---|
| advective (current) | ~1.0 x mdot | 0.997 | 6.2 |
| conservative | 15-60 x mdot | 0.933 | 16.7 |

MULES limits the antidiffusive flux to keep alpha bounded *assuming* the
advective update form. Change the form and the boundedness guarantee is lost, so
alpha overshoots and the balance degrades. The missing terms must therefore
enter **before** `mules_limit!` computes its bounds - which is the same
restructure `apply_phase_change_alpha!` already flags as necessary for vigorous
boiling. The two cannot be done independently, and neither is a small change.

### Implicit alpha transport — implemented, opt-in, measured WORSE

`Mixture(alpha_transport = :implicit)` solves a conservative volume-fraction
equation as a linear system through `solvers.alpha`, instead of the explicit
MULES update. Sources enter the matrix, so the limiter-ordering constraint that
blocked the conservative fix does not exist, and the alpha-Courant limit
disappears.

**Measured on the pipe case, and it is worse:**

| alpha transport | vapour mass residual | alpha_min | max\|U\| |
|---|---|---|---|
| `:mules` (default) | ~1.0 x mdot | 0.997 | 6.2 |
| conservative, explicit | 15-60 x mdot | 0.933 | 16.7 |
| `:implicit` | 3-30 x mdot, GROWING with step | 0.911 | 17.2 |

So **all three attempts at the alpha equation made the measured balance worse**,
which is worth recording as a result in itself: it suggests either the
conservative form is not the right target, or the diagnostic is not measuring
what it appears to.

Caveat on the comparison: `alpha_fluxf` is rebuilt differently on the implicit
path (`mdotf*alphaf`) than MULES produces, so the vapour flux entering the
diagnostic is not strictly like-for-like between the two rows.

**The default is therefore unchanged (`:mules`).** The implicit path is kept
because it is a defensible formulation, it is regression-clean, and it removes
the Courant limit — but it should not be selected without re-measuring.

**Side benefit, independent of the above:** `solvers.alpha` was previously
accepted and silently ignored, and the reported alpha residual was a hard-coded
`0.0`. On the implicit path both are now real — the mixture regression reports
`alpha: 1.09e-16` rather than a placeholder, so there is finally convergence
feedback on the volume fraction.

### NOT SOLVED — near-wall cooling persists

**Status: open.** The near-wall cell cools below the inlet temperature and the
case eventually explodes, under every configuration tried, including with AMG
(which the user reports is the most stable option over long runs).

**A caution about the evidence in this document.** Most of the eliminations below
were made from 15-60 step runs. That window is long enough to distinguish
"diverging fast" from "diverging slowly", but **not** long enough to establish
stability - a point demonstrated by the DILU result in the next section, which
looked conclusive at 15 steps and did not hold over a longer run. Where a
conclusion below says "stable", read it as "not yet diverging at N steps".

Relative comparisons (same solver, same duration, one variable changed) remain
sound. Absolute claims of stability do not.

**What is solid**, independent of the open problem:

- the RPI flux partition inverts exactly (`q_total = q_w` to 5e-13 on every face)
- the wall superheats correctly (`T_wall` 31.6-35.2 K at `q_w` = 3e4 W/m^2)
- the wall energy balance is net heating: +20149 W/m^2 of 30000 delivered
- the coldest cell has `S_T = 0` - no latent sink at all - so the cooling is not
  produced by the evaporation source term
- 53% of the domain falls below inlet temperature while the sink exists in only
  3.4% of cells, so the cold is transported, not generated locally

**What has been fixed along the way** (all verified, all regression-clean):
`p_abs_prev` seeding, the Mixture energy-flux inconsistency, the `PropertyGrid`
constructor shadowing, the `KocamustafaogullariIshii` contact-angle units, and
the `HibikiIshii` saturation behaviour.

**Approach note.** Repeated short-run bisection produced several confident
diagnoses that were later disproved (psi*dp/dt, pressure work, thermal
expansion, the volume source, `make_symmetric!`, and DILU). The remaining
problem probably needs long runs with per-step field monitoring rather than more
single-variable elimination at 15-60 steps.

### Linear convergence measurement — Jacobi vs DILU (15 steps only)

Found by logging `Krylov.iteration_count` and the returned residual per call.

| preconditioner | residual | p_rgh range | max\|U\| | min T | calls hitting itmax |
|---|---|---|---|---|---|
| Jacobi | 3.2e-7 .. 7.5e-6 | -602169 .. -2159 | 7.46 | 28.487 | **45 of 45** |
| **DILU** | **1.1e-10** .. 1.4e-7 | **58 .. 12338** | **5.92** | **28.929** | 40 of 45 |

With `itmax = 1000` and a target of `atol + rtol*norm(b)` ~ 3.5e-10, Jacobi
stalled **3-4 orders short**, at a relative residual near unity — it was barely
reducing the error at all, and returning that as the pressure field.

The wall cells are 34 um x 1.5 mm x 0.4 mm, an aspect ratio of ~44:1. A pressure
Poisson system on that stretching is beyond Jacobi.

**Everything downstream followed from this**: near-wall cells colder than the
inlet, `max|U|` far above the inlet value, `p_abs` driven outside the property
tables, and the runaway through the vapour density (`expansion ~ 1/rho_v`, and
`rho_v` collapsed to 2.61 against a correct 8.82 once `p_abs` fell to ~0.3 MPa).

**It also failed with wall boiling OFF** (44 of 45 calls). Those runs only looked
healthy because the un-converged iterate happened to be close enough. **Any
earlier result in this document obtained on this mesh with Jacobi for `p_rgh`
should be treated as unconverged**, including the source-term eliminations —
they remain valid as relative comparisons but not as absolute statements.

If trying AMG, note its default `itmax` is 200, well below what this system
needs; it will fail the same way and look identical from the outside.

### ROOT CAUSE (1) — an explicit thermo-acoustic loop, over its CFL limit

Two terms exist only on the compressible path and form a closed loop:

```
dT -> expansion = beta*dT/dt -> dp -> dp/dt -> S_T = beta*T*dp/dt -> dT
```

Each traversal divides by `dt` twice. Treating it explicitly therefore carries an
**acoustic CFL condition**, `dt < dx/c` with `c = 1/sqrt(rho*psi)`. For liquid
hydrogen here `c ~ 420` m/s and the wall cells are 34 um, giving `dt < 8e-8` s.
The case needs `dt ~ 2e-6` — **25x over the limit**.

Confirmed by disabling both legs simultaneously (neither alone suffices):

| configuration | max\|U\| | T |
|---|---|---|
| both live | NaN / 1e39 | pinned at solver limits |
| `pressure_work_relax = 0.5` | 4.1e39 | pinned |
| `pressure_work_relax = 0.05` | 1.9e37 | pinned |
| **both zeroed** | **7.67 m/s** | **[28.97, 29.48] K** |

This also finally explains the one clue that never fitted: `dt = 1e-8` gives
acoustic CFL ~0.12 and was nearly stable (max\|U\| = 13).

And why K-Site never hits it: its liquid is `ConstEos` (`psi_l = 0`), it is
sealed, and `dp/dt` is order 1 Pa/s rather than 5.6e7 — the loop is never
excited.

**Resolution.** `pressure_work_relax` and `expansion_relax`, both accepting
`[0, 1]` where **0 disables the term**. Justified rather than a fudge: both terms
vanish at steady state, so a steady-state answer is unaffected. What is given up
is the transient thermo-acoustic response — which a self-pressurising tank needs
and must therefore leave on, but a forced-convection heat transfer case does not.

`expansion_relax` damps only the thermal part; the phase-change volume creation
is summed in afterwards and always survives at any relaxation.

### Superseded — `Dp/Dt` alone is not the whole story

Fixing the seeding moved the problem to step 2, where the first pressure solve
establishes the whole ~618 Pa frictional drop from `p_rgh = 0`. That transient
physically propagates acoustically over `L/c ~ 7e-4` s (~350 steps); a segregated
pressure solve compresses it into one, so `Dp/Dt` is ~350x too large.

This cannot be fixed by better initialisation: with `beta*T ~ 1.6` at
`dt = 2e-6`, **a 1 Pa pressure adjustment produces `S_T ~ 8e5` W/m^3**.
Initialising `p_rgh` to the developed frictional profile does not help (tested).

With `update_pressure_work!` disabled the case no longer produces NaN, but the
pressure solve hits its iteration limit — slow and ill-conditioned rather than
explosive.

**Implemented:** `Fluid{Multiphase}(..., pressure_work_relax = 0.5)` applies
[`relax_source!`](src/Solvers/Solvers_5_Multiphase.jl) to `dpdt`. **On by
default** at 0.5, matching STAR-CCM+'s 50/50 body-force blending (Eq. 2923);
`1.0` disables it. Safe for the sealed-tank cases because blending preserves the
converged value — a slowly-varying `dp/dt` is recovered to 0.1 % after ten steps
— and all 3322 regression tests pass, including the compressible-ullage suite
that pins K-Site's dp/dt to 0.09 %.

**It damps the pipe divergence but does not cure it.** Measured at q = 0,
40 steps, otherwise identical:

| `pressure_work_relax` | max\|U\| |
|---|---|
| 1.0 (off) | 2.1e56 |
| 0.5 (default) | 4.1e39 |
| 0.05 | 1.9e37 |

Many orders of magnitude of damping, but still exponential growth. Consistent
with the analysis: blending cuts the *peak* of the spurious spike by roughly
`relax`, while the underlying feedback loop is untouched. Note also that T
collapses onto the lower solver limit (19 K) in the 1.0 and 0.5 runs, i.e. the
pressure-work term still dominates the energy equation.

**Remaining work.** Damping `Dp/Dt` is treating a symptom. The options are to
make the pressure-work source switchable off entirely for stiff through-flow
cases, or to address why the pressure solve is so ill-conditioned on this mesh —
with the term disabled the run stops producing NaN but the linear solver hits its
iteration limit, so there is at least one conditioning problem behind it.

### Earlier hypotheses — all tested and eliminated

An earlier draft of this note attributed the failure to the `psi·dp/dt` term.
**That was tested and disproved** — it is recorded here because the reasoning is
still useful, but it is not the answer.

The `psi` hypothesis was: near-critical liquid hydrogen at 0.7 MPa has
`psi_l ≈ 1.0e-7` 1/Pa, about **220× that of water** (4.5e-10) — a correct
property value, confirmed against the EOS, not a table artefact — and with the
618 Pa frictional drop appearing in the first step this gives
`1.0e-7 × 618 / 2e-6 ≈ 30 s⁻¹` of spurious volumetric source.

Plausible, but wrong: building the tables with `RealFluid(..., p_ref = p_sat)`
makes density independent of pressure and sets `psi` to **exactly zero**, and
the case still diverges. So `psi·dp/dt` is at most a contributor.

### What has been ruled out

| Hypothesis | Test | Result |
|---|---|---|
| The boiling models | disable Lee and RPI entirely | still diverges — **not** it |
| Heating | q = 0 | still diverges — **not** it |
| Through-flow BCs / mesh / wall functions | same case, `ConstEos` phases | **stable**, 5.34 m/s — not it |
| Velocity-inlet + pressure-outlet unsuited to compressible flow | the repo's validated subsonic compressible case (`2D_cylinder_heated_unsteady`, CPISO) uses the identical arrangement | not indicated |
| `psi·dp/dt` | `p_ref` locking, `psi ≡ 0` | still diverges — **not** it |
| Startup shock from `p_rgh = 0` | initialise to the developed frictional profile | still diverges (1.2e4 m/s) |
| Time-step too large | dt 2e-6 → 1e-8 (200×) | largely suppressed (13 m/s) |

The isolation is nonetheless sharp: **identical mesh, BCs, turbulence model and
inlet velocity, differing only in whether the phases carry a variable equation of
state.** So the fault is on the compressible branch of `MULTIPHASE`, which is
taken whenever `is_compressible_multiphase` is true.

### Source terms individually eliminated

Each of the compressible branch's source terms was disabled in turn, by
overriding its function from the test script. **None of them is the cause.**

| Term disabled | How | Result |
|---|---|---|
| `psi*dp/dt` | `RealFluid(..., p_ref=p_sat)` → `psi ≡ 0` | still diverges |
| Pressure work `S_T = beta*T*dp/dt` | override `update_pressure_work!` to fill zero | still diverges |
| Thermal expansion `beta*dT/dt` | override `update_expansion!` to fill zero | still diverges (7.6e6 m/s) |
| Missing `make_symmetric!` | add it to `solve_pressure_compressible!` | **no change at all** — identical max\|U\| to baseline (521522.57573746226) |
| Stiff phase-change sources | `phase_change_relax` / `wall_boiling_relax` swept 1.0 → 0.1 → 0.01 | identical divergence at every value |

The relaxation sweep is the strongest single result: at `relax = 0.01` both
vapour sources are effectively switched off, and the case diverges *identically*
to `relax = 1.0`. Consistent with — and independently confirming — the earlier
finding that removing both boiling models entirely changes nothing.

The `make_symmetric!` result is worth keeping in mind separately: the
incompressible pressure equation gets it (single Laplacian term), the
compressible one never does — neither through `solve_equation!`'s
`length(terms) == 1 && terms[1] isa Laplacian` guard, nor in
`solve_pressure_compressible!`. Adding it is *legitimate* (a Time term
contributes only to the diagonal, so all off-diagonals still come from the
Laplacian) but it changes nothing here, so the matrix was evidently already
symmetric.

### What is left

With `psi = 0`, expansion zeroed and pressure work zeroed, and no phase change,
the compressible branch should be numerically almost identical to the
incompressible one. Only two differences remain:

1. **`update_phase_state!` refreshes properties every step**, so `rho = rho(T)`
   varies — and with it `snGrad(rho)`, which `phi_gf!` and
   `well_balanced_pressure_grad!` both build the buoyancy term from. These are
   well balanced by construction only when `rho` is piecewise constant. This is
   exactly what §5.4 of the sealed-tank plan predicted as "the item most likely
   to cost unplanned time in step 5", and it is now the leading suspect by
   elimination.
2. **`solve_pressure_compressible!` is used instead of `solve_equation!`**,
   though with `psi = 0` its frozen `p_rgh_start` reference contributes nothing.

**Decisive test still to run:** all three sources zeroed *simultaneously*, with
q = 0. If that is stable, the instability is a feedback loop that no single
source disable can break (each elimination above changed only one leg of it). If
it still diverges, the buoyancy discretisation under variable `rho` is the
cause, and §5.4 is the item to work on.

### Constraint on any fix

The sealed-tank behaviour is validated (dp/dt exact to 0.09%) and must not
regress. Also worth revisiting: the `pref` prohibition on the compressible path
assumes a sealed domain, and should probably become conditional.

### Usable today

Setting both phases to `ConstEos` makes the case stable immediately, at the cost
of the real-gas compressibility. That is enough for the single most valuable
first validation — the **non-boiling Dittus-Boelter branch** (the paper's own
first check, stated in its Conclusion), which needs no compressibility at all and
isolates the mesh, wall functions and turbulence model from every boiling
closure. It should be done before any boiling comparison regardless.

---

## What was built

### Real-fluid properties — `RealFluid`

- [`src/ModelPhysics/2_tabulated_properties.jl`](src/ModelPhysics/2_tabulated_properties.jl):
  `PropertyGrid`, `PropertyTable`, `table_lookup` (bilinear, kernel-safe,
  clamped outside the grid), the five property models `TabulatedEos`,
  `TabulatedMu`, `TabulatedK`, `TabulatedCp`, `TabulatedBeta`, and
  `SaturationCurve`.
- [`src/ModelPhysics/FluidProperties/property_tables.jl`](src/ModelPhysics/FluidProperties/property_tables.jl):
  the host-side builder walking the Helmholtz H2/N2 EOS.

Tabulation is a requirement, not an optimisation: a direct Helmholtz evaluation
root-finds and allocates ~20 vectors per call, so it can neither run in a
`KernelAbstractions` kernel nor be afforded per cell per step.

**Why it is needed at all.** At the paper's operating pressures the vapour is at
up to 85% of the critical pressure, and the real compressibility departs from
ideal by far more than a modelling tolerance:

| p | psi_real / psi_ideal |
|---|---|
| 0.4 MPa | 1.4× |
| 0.7 MPa | 1.9× |
| 1.1 MPa | 4.4× |

Table lookups reproduce the direct EOS at the 0.7 MPa saturation point:

| | table | direct EOS |
|---|---|---|
| rho_l | 56.7536 | 56.75 |
| rho_v | 8.8353 | 8.82 |
| cp_l | 24604 | 24609 |
| beta_l | 0.05499 | 0.05492 |
| mu_l | 6.988e-6 | 6.99e-6 |
| k_l | 0.09526 | 0.0953 |

**Off-branch states.** The mixture blend evaluates *both* phases in *every* cell,
so a single-phase liquid cell still asks for a vapour density. Where the
requested branch has no root, the branch is continued metastably within 5 K of
saturation and otherwise falls back to the saturation line — mirroring what
`EOS_wrapper` already does.

### RPI wall boiling

[`src/ModelPhysics/2_wall_boiling_models.jl`](src/ModelPhysics/2_wall_boiling_models.jl)
(physics) and [`src/Solvers/Solvers_5_wall_boiling.jl`](src/Solvers/Solvers_5_wall_boiling.jl)
(solver coupling).

`q_w = q_c + q_q + q_e`, with four independently swappable closures:

| Supertype | Implementations |
|---|---|
| `AbstractNucleationSiteDensity` | **`LemmertChawla`** (default), `HibikiIshii` |
| `AbstractDepartureDiameter` | **`TolubinskyKostanchuk`** (default), `KocamustafaogullariIshii` |
| `AbstractDepartureFrequency` | **`Cole`** (default) |
| `AbstractInfluenceArea` | **`DelValleKenning`** (default), `ConstantInfluenceArea` |

Adding a model means declaring a struct under the relevant supertype and giving
it one method. All four receive the same `BoilingState`, so a new correlation can
use any local quantity without changing a signature anywhere.

**Coupling.** RPI does *not* replace the temperature boundary condition. The
`FixedHeatFlux` wall still delivers the whole `q_w`; RPI adds the vapour source
`mdot_wall = q_e·A/(h_fg·V)` into the *same* `mdot_pc` field the bulk models
write. The existing latent-heat sink then removes `q_e·A` again, so the liquid is
net-heated by `q_c + q_q` without the BC or the energy equation knowing that
boiling is happening. Sharing the field is what lets either mechanism run alone.

**Wall temperature.** The experiment prescribes heat *generation*, so the flux is
known and `T_w` must be solved for. `solve_wall_temperature` bisects on a bracket
that is guaranteed by construction (`T_sat` below, the pure-convection wall
temperature above). Bisection rather than Newton because `N_a ~ dT_sup^1.805` is
extremely stiff near onset, and because a fixed iteration count is branch-free.

### Source under-relaxation

`Fluid{Multiphase}(..., phase_change_relax = 1.0, wall_boiling_relax = 1.0)` —
independent temporal damping of the two vapour sources, because they are stiff
for different reasons (bulk: `(T - T_sat)` across the interface; wall:
`N_a ~ dT_sup^1.805`, far steeper).

Implemented as **blending, not scaling**:

```
mdot = (1 - relax)*mdot_prev + relax*mdot_new
```

Scaling would be the obvious reading of "under-relaxation" and is wrong here: it
would permanently evaporate less mass than the model asks for, biasing the mass
balance. Blending damps only how fast the source may *change* — at steady state
`mdot == mdot_prev`, so the relaxed and unrelaxed answers coincide. A unit test
pins exactly this (200 iterations at a fixed raw rate must converge to the raw
rate, not to `relax*raw`).

Applied to the bulk rate *before* the wall contribution is summed into
`mdot_pc`, so the bulk factor does not leak onto the wall source.

Defaults of 1.0 reproduce the previous behaviour exactly.

### Solver generalisation

- Per-phase **face** fields for cp, k and mu (`PhaseFaceProperties`), mirroring
  what already existed for density. This is what lifted the constant-cp/k
  restriction in `TwoPhaseTemperature`.
- `beta` is now read as an indexable field rather than a scalar, so a tabulated
  expansivity varies per cell.
- Turbulent two-phase conductivity `keff += (rho·cp)_f·nu_t/Pr_t`, replacing the
  error that previously refused any non-laminar closure.
- `Lee`/`Schrage` now accept any EOS carrying a specific gas constant
  (`specific_gas_constant`), not just `IdealGas`.

### Mesh

[`examples/0_GRIDS/lh2_pipe_sector/`](examples/0_GRIDS/lh2_pipe_sector/) — 90°
butterfly O-grid, first cell height *solved* for a target y+ rather than guessed.
Achieved y+ 33–48 for the default case. An O-grid rather than a wedge because a
wedge collapses onto the axis and leaves `reconstruct!` rank-deficient; measured
here at 7.1e-15 uniform-field error.

---

## Defects found in existing code

1. **`PropertyGrid` constructor shadowing** (introduced and fixed in this work).
   The stored layout `(p_min, dp, np, …)` has the same arity and types as the
   natural `(p_min, p_max, np, …)` spelling, so a positional outer constructor is
   shadowed by the compiler-generated one and `p_max` is silently taken as the
   spacing. Now keyword-only. Worth remembering as a general Julia hazard.

2. **`params_computation` unit conversion** — `conversion_factor = 1/(M*1e3)`
   is applied to `cp`, `cv`, `u`, `h` and `s`. Converting J/mol to J/kg requires
   `1/M`, so these come out **1000× too small**; liquid H2 `cp` would read
   ~19 J/kg/K instead of ~19000. It is additionally applied to `beta`, which is
   already intensive (1/K) and must not be rescaled at all — flagged
   `NOT TESTED!!!!` in that function.

   **Not fixed here**, because `property_tables.jl` bypasses it entirely (it
   derives mass-specific values directly from the molar ones). It remains live
   for any other caller of `params_computation` / `EOS_wrapper`. The existing
   `unit_test_fluidProperties.jl` does not cover `cp`, which is why it has gone
   unnoticed.

3. **`EOS_wrapper` / `HighFidelity_Closure` tuple ordering.** `EOS_wrapper`
   destructures `params_computation` with `cp` and `cv` transposed, and returns
   `beta` and `entropy` in positions that `HighFidelity_Closure` reads in the
   opposite order. The cp/cv swap happens twice and cancels; **beta and entropy
   do not** and are returned transposed. Also not fixed, same reasoning.

---

## Test coverage added

| File | Tests |
|---|---|
| `test/unit_test_wall_boiling.jl` | 95 |
| `test/unit_test_property_tables.jl` | 124 |
| `test/unit_test_lh2_pipe_sector_mesh.jl` | 28 (skips if the mesh is not built) |

No regression: `unit_test_twophase_energy`, `unit_test_compressible_ullage`,
`unit_test_phase_change`, `unit_test_phase_properties` (3334 tests) and the three
multiphase solver cases all pass.

---

## Known gaps

- **Wall boiling diagnostics are not reachable.** `WallBoilingState` carries
  `T_wall`, `q_evap`, `q_quench` and `q_conv` per wall face — the partition is
  the whole substance of the model — but it is local to `MULTIPHASE` and is
  neither returned nor written to VTK. Comparing against the paper's boiling
  curve needs `T_wall`, so this has to be exposed before step 8.
- **Coefficients are water-fitted.** `LemmertChawla` (m = 210, n = 1.805) and
  `TolubinskyKostanchuk` (0.6 mm, 45 K) have no established cryogenic values.
  Expect recalibration to be part of validation, not a sign of a bug.
- **No DNB criterion.** RPI models nucleate boiling only. The paper's DNB
  correlation (Eqs. 1–5) is out of scope, and the `alpha_min` ramp in `RPI` is a
  numerical safeguard, *not* a dryout model.
- **Experimental data not in the repo** — would need digitising from the paper's
  figures.

---

## Mass-form pressure equation (`pressure_form = :mass`)

**Status: implemented, opt-in, verified on the sealed-ullage regression.
NOT yet run on the pipe case.**

### Why

XCALibre's pressure equation was VOLUME-based:

    psi*dp/dt - div(rDf grad p_rgh) = -div(u*) + expansion      [1/s]

The pressure correction therefore constrains the *volumetric* flux. But momentum
and energy convect with `rhoPhi`, a *mass* flux, and nothing in the volume form
ever enforces `div(rho u) = -drho/dt`. With `rho_l/rho_v ~ 57` the two fluxes are
nowhere near proportional. This is what the measured `rel = 1.0` on
`d(rho)/dt + div(rhoPhi)` was — present in **every** case, including the
constant-density control.

STAR-CCM+ solves the MASS form, which is the same statement multiplied through by
`rho_m` (`u.grad(rho_m)` is absorbed because `Drho_m/Dt + rho_m div(u) = 0`):

    drho_m/dp*dp/dt - div(rho_f rDf grad p_rgh) = -div(rho_f u*) + rho_m*expansion

### Term mapping

| term | `:volume` | `:mass` |
|---|---|---|
| Time flux | `sum_i a_i psi_i` | `sum_i a_i rho_i psi_i` (`= drho_m/dp`) |
| Laplacian flux | `rDf` | `rho_f*rDf` |
| RHS divergence | `div(u*)` | `div(rho_f u*)` |
| thermal expansion | `sum_i a_i beta_i DT/Dt` | `rho_m *` that |
| phase change | `mdot*(1/rho_v - 1/rho_l)` | `rho_m *` that |

Every term picks up the density of the quantity it belongs to. That pattern is
the check that the conversion is right.

**Two corrections made while deriving this**, both worth recording because the
first was wrong in a plausible-looking way:

1. I first expanded `d(rho_m)/dt` into composition terms, got
   `rho_l*(-mdot/rho_l) + rho_v*(+mdot/rho_v) = 0`, and concluded there is no
   phase-change source at all. **Wrong** — that expansion silently drops the
   dilatation, which is the very thing the pressure equation exists to produce.
   Circular. The scale-by-`rho_m` reading is the honest one, and it matches the
   STAR-CCM+ source term verbatim:
   `b_cell = sum_f [mdot_lv*(rho_m_f/rho_v - rho_m_f/rho_l)]*V_cell`.
2. The time-term coefficient is `sum_i a_i rho_i psi_i` and **not**
   `rho_m*sum_i a_i psi_i`. The scale-by-`rho_m` reading is exact for fluxes and
   sources, but the time coefficient is a genuine derivative `drho_m/dp` and each
   phase must carry its own density. They coincide when one phase dominates and
   differ by ~2x at `alpha = 0.5`.

### The implementation trap

`correct_mass_flux_mp!` reads the correction off the assembled matrix, which is
now built from `rho_f*rDf` — so it is a MASS flux correction. Adding it to a
volumetric `mdotf` would be wrong by ~57x, and every consumer downstream
(`alpha_fluxf`, the alpha equation, the Courant numbers) would inherit it
silently.

Handled by scaling `mdotf` to a mass flux *before* the divergence and back to
volumetric *after* the correction, rather than by scaling the correction. This
also means the shared SIMPLE kernel needs no change, and the boundary correction
(which SETS rather than adds on some patches) lands on the right quantity either
way.

`rDf` needs its own storage in mass form: the equation's flux array now holds
`rho_f*rDf`, but `phi_gf!`, `pressure_grad!` and `correct_velocity_rgh!` all
still want the plain `rDf` — the velocity correction is `-rD grad(p_rgh)`
regardless of how the pressure equation was scaled, because the momentum equation
is untouched.

### Measured

Sealed ullage, `unit_test_compressible_ullage.jl`:

| | `:volume` | `:mass` |
|---|---|---|
| `dp/dt` vs exact `R*Q/(V*cv)` | 0.98989 | 0.98977 |
| vapour mass drift over 200 steps | 8.11e-6 | **4.86e-9** |

The exact analytical `dp/dt` is reproduced (it does not depend on how the
equation is scaled, which is what makes it a valid acceptance test), while the
mass drift falls by a factor of **1670**. That is precisely the signature
predicted: the mass form conserves mass directly.

The two forms agree to 1e-6 on pressure and 1e-5 relative on temperature. They do
**not** agree to solver precision, and should not: `rho_f` varies over the mesh
and the time coefficient is not a uniform rescale. An earlier version of the test
asserted 1e-6 on temperature and failed at 2.0e-4 K on a 20.4 K field — the
assertion was wrong, not the code.

### Still open

- **Not yet run on the pipe case.** Enabled in
  `test/0_TEST_CASES/3d_LH2_pipe_forced_convection.jl`. Acceptance test is the
  existing mass-conservation diagnostic: baseline `|div| = 6.75e4, rel = 1.0`
  with boiling, `|div| = 84, rel = 1.0` on the control.
- **`blend_rhoPhi!(::Mixture, ...)` is still `mdotf*rhof`**, a plain product,
  where the VOF branch builds `rhoPhi` consistently from the alpha flux. The mass
  form now produces a *corrected* mass flux inside the pressure loop, so feeding
  that directly into `rhoPhi` is the obvious next step — deliberately left for a
  separate change so its effect can be measured on its own.
- **The constant-density control fails at `|div| = 84`, where mass and volume
  conservation are identical.** That failure is purely the linear solve (45 of 45
  pressure solves hitting `itmax`) and the mass form cannot fix it. It sits
  underneath everything else and should be fixed first; a converged volume-based
  solve may well turn out to be adequate.
