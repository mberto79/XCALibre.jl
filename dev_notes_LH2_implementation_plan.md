# LH2 tank self-pressurisation — implementation plan

Companion to `dev_notes_LH2_tanks.md` (gap analysis) and
`test/0_TEST_CASES/3d_LH2_ksite_selfpressurisation.jl` (target case).

Decisions taken: **OpenFOAM wedge mesh via blockMesh**, **full EOS ullage**
(ρ_v = ρ_v(p,T) from the Helmholtz H2 EOS + ψ ∂p/∂t), all three phase change
models exposed in the user interface.

Nothing below has been implemented. Steps 1–2 and 4 are self-contained; steps
3, 5 and 6 are substantial changes to `Solvers_5_Multiphase.jl`.

---

## Status

| Step | State |
|---|---|
| 1. K-Site wedge mesh | **done** — generated, meshed, geometrically validated |
| 2. Phase thermal properties | **done** — 41 unit tests, no regression |
| 3. Two-phase energy equation | **done** — `Energy{TwoPhaseTemperature}`, 12 tests |
| 4. Fixed-heat-flux BC | **done** — `FixedHeatFlux`, exact energy balance to 7e-11 |
| 5. Compressible ullage | **done** — ideal gas + psi dp/dt + expansion; dp/dt exact to 0.09% |
| 6. Phase change models | **done** — Schrage / MeJ / Lee + Antoine, 59 tests |
| 7. Post-processing | **partial** — `ullage_average`, `liquid_volume`, `boil_off_rate` |
| 8. Validation matrix | pending — needs experimental p(t) traces |

**Step 0 is closed — the paper is now in hand** (Fernandes, Korsukova, Ellis,
Ambrose & Eastwick, IJHMT 256 (2026) 128067), which settles both open items and
corrects several assumptions made before it was available:

- **Geometry.** There was never an inconsistency. The paper gives major diameter
  2.20 m *and minor diameter 1.93 m* (ratio 1.14, not the 1.2 taken from a
  secondary source). With a = 1.100 m, b = 0.965 m the reported volume
  (4.89 m³) and surface area (13.98 m²) both follow exactly, and as an
  independent check the flux × area heat loads reproduce the paper's 49.0 W and
  28.0 W. The mesh generator has been corrected.
- **Cases.** K(i) and K(ii): both **50 % fill** (so the interface is at z = 0
  exactly), initial pressure **103 kPa**, wall flux 3.50 and 2.00 W/m²,
  durations 17.5 and 20.0 hr.
- **Energy.** The paper solves its energy equation **for temperature** — which
  independently confirms the step 3 design decision taken before it was
  available.
- **Surface tension is neglected** ("several orders smaller than the other
  terms"), so `VOF(sigma=0.0)`.
- **Laminar**, deliberately: RANS closures reproduced the pressure rise and
  vapour stratification *less* well.
- **Vapour is an ideal gas**, properties temperature-dependent from NIST. This
  simplifies step 5 considerably — a full Helmholtz ullage is not what the
  reference does.
- **Conjugate heat transfer** through the 2.08 mm wall, flux applied at the
  *outer* solid surface. Fig. 6(c) shows this redistributes heat towards the
  interface, so the flux into the fluid is markedly non-uniform. Applying the
  flux directly to the fluid is a real simplification, now recorded in the case
  file.

Still to obtain: the digitised experimental p(t) traces (paper Fig. 7) and the
measured initial temperature profile for the K-Site cases.

---

## Step 0 — Two things to resolve before coding

**0.1 The paper.** I could not access the full text (Elsevier 403,
ResearchGate 403, NASA PDFs are scanned images). The exact formulations of the
three models, the three coefficients swept per model, and the paper's initial
conditions are all unconfirmed. The formulations in step 6 are the standard
literature forms. If they differ from the paper's, step 6 changes but nothing
else does.

**0.2 The geometry discrepancy.** A pure ellipsoid with 2.2 m major diameter
and 1.2 axis ratio gives V = (4/3)πa²b = **4.645 m³**, but K-Site is reported as
**4.89 m³** — a 5 % difference. Likely a cylindrical mid-section. This must be
settled before the mesh is built, because it changes the ullage volume and
therefore the pressurisation rate directly.

---

## Step 1 — K-Site wedge mesh

**Deliverable:** `examples/0_GRIDS/ksite_wedge_5deg/` (blockMeshDict + generated
polyMesh), loaded with `FOAM3D_mesh`.

**Geometry:** 5° wedge of the half-ellipse in (r,z), a = 1.1 m, b = 0.9167 m,
revolved about the z axis. One cell thick in the azimuthal direction, wedge
planes as `Symmetry`.

**Topology.** A naive single block from axis to wall gives degenerate cells at
r = 0 (zero-area faces, prism-shaped "hexes"). OpenFOAM tolerates this; **XCALibre
may not**. Recommended topology is therefore a 2-block arrangement in (r,z):

- an inner rectangular core block, r ∈ [0, r_c], z ∈ [−z_c, z_c]
- graded blocks from the core to the elliptical wall (arc edges)

which still has an axis edge but confines the degeneracy to one block face and
keeps wall-normal resolution controllable. Wall-normal grading is needed — the
thermal boundary layer under 3.5 W/m² is thin relative to a 1.1 m tank.

**Risk (must be tested first, in isolation):**
1. Does `FOAM3D_mesh` build valid connectivity/geometry for wedge cells with a
   collapsed axis edge? Check `faces[i].area`, `delta` and `normal` for
   near-axis faces — a zero `delta` will produce `Inf` in every `snGrad`
   kernel in the multiphase solver.
2. Does `Symmetry` behave correctly on the two non-axis-aligned wedge planes?
   `src/Discretise/boundary_conditions/symmetry.jl` and its `_interpolation`
   sibling need reading against a wedge normal.
3. Does the reconstruction in `reconstruct!`
   ([Solvers_5_Multiphase.jl:719](src/Solvers/Solvers_5_Multiphase.jl#L719))
   stay well conditioned? It inverts a 3×3 face-normal moment matrix per cell;
   with a one-cell-thick wedge the azimuthal direction is nearly singular and
   it falls back to `invdet = 0`, which would silently zero the pressure
   gradient reconstruction. **This is the highest-risk item in step 1** and
   should be checked before anything else is built on top of the wedge.

**Validation gate:** re-run the existing hydrostatic test
(`2d_multiphase_hydrostatic.jl`) physics on the wedge mesh — a static LH2/GH2
column must not develop spurious currents. If item 3 above bites, this is where
it shows up.

**Fallback if the wedge fails:** full 3D sector (e.g. 30°) with an O-grid core,
which removes the axis degeneracy and the ill-conditioned reconstruction at the
cost of more cells.

---

## Step 2 — Phase thermal properties

**Files:** `src/ModelPhysics/2_thermophysical_models.jl`,
`src/ModelPhysics/2_fluid_models.jl`.

`PhaseState` already carries `k`, `cp` and `beta`
([2_fluid_models.jl:229](src/ModelPhysics/2_fluid_models.jl#L229)) and
`build_phase` allocates them, but `Phase(; rho, mu)` does not accept them and
nothing in the codebase reads them.

1. Add `ConstK`, `ConstCp`, `ConstBeta` model types mirroring the existing
   `ConstEos` / `ConstMu` pattern
   ([2_thermophysical_models.jl:27](src/ModelPhysics/2_thermophysical_models.jl#L27)),
   with `AbstractThermalConductivityModel`, `AbstractHeatCapacityModel`,
   `AbstractExpansivityModel` supertypes.
2. Extend `Phase(; rho, mu, k=nothing, cp=nothing, beta=nothing)` with the same
   "bare float gets wrapped in a Const* model" promotion already used for
   `rho` and `mu`.
3. Extend `build_phase` to allocate `ConstantScalar` vs `ScalarField` per model
   type, as it already does for `rho`/`mu`.
4. Add `update_phase_properties!(phase, p, T, config)` dispatching on the model
   types — no-op for the `Const*` variants, EOS lookup for the Helmholtz ones.

**Validation gate:** unit test that a `Phase` with all five properties round-trips
into a `PhaseState` with the right field types, on CPU and GPU (`isbits` check —
this is where a non-`isbits` property model would break GPU dispatch, cf. commit
`a9eb4b33` which fixed exactly that class of bug in the viscosity framework).

---

## Step 3 — Two-phase energy equation

**Files:** new `src/ModelPhysics/Energy/multiphase_energy.jl`;
`src/Solvers/Solvers_5_Multiphase.jl`.

**Design decision: solve temperature directly, not enthalpy.**

The existing `SensibleEnthalpy` path solves `he` with `h = cp(T − Tref)`. Across
a VOF interface `cp` jumps by ~26 % (9660 → 12200 J/kg/K), so `h` is
*discontinuous* even where `T` is continuous. Solving `h` and recovering
`T = h/cp + Tref` would smear that discontinuity across the interface and
corrupt exactly the quantity the phase change models depend on — `(T − T_sat)`
at the interface. Reference two-phase codes solve `T`. So:

```
Time{...}(rho_cp, T) + Divergence{...}(rho_cp_phi, T) - Laplacian{...}(keff, T)
    == -Source(mdot_hfg) + Source(S_compress)
```

with

- `rho_cp   = α ρ_l cp_l + (1−α) ρ_v cp_v`
- `keff     = α k_l + (1−α) k_v  (+ ρ cp ν_t/Pr_t when turbulent)`
- `rho_cp_phi` the volumetric flux weighted by the same blend — **must be
  built from the same `alpha_fluxf` that `blend_rhoPhi!` uses**
  ([Solvers_5_Multiphase.jl:520](src/Solvers/Solvers_5_Multiphase.jl#L520)),
  otherwise energy and mass advection are inconsistent at the interface and the
  interface temperature drifts.
- `S_compress` the ullage pressure work (step 5); zero until then.

**API:** new `Energy{TwoPhaseTemperature}(; Tref)` in `energy_types.jl` +
`multiphase_energy.jl`. This leaves the single-phase `SensibleEnthalpy` /
`InternalEnergy` paths completely untouched.

**Dispatch fix (required):** `run!` currently routes on `F<:Multiphase` alone
([Solvers_3_solver_dispatch.jl:48](src/Solvers/Solvers_3_solver_dispatch.jl#L48)),
so any energy model is silently accepted and ignored. Either add an explicit
method for `E<:TwoPhaseTemperature`, or — better — make the existing method
error on any `E` that is neither `Isothermal` nor `TwoPhaseTemperature`. Silent
acceptance of an ignored energy model is a trap worth closing regardless of
this project.

**Solver-loop placement:** in `MULTIPHASE`, after `update_mixture_properties!`
and before the pressure correctors, mirroring where `energy!` sits in `CPISO`
([Solvers_2_CPISO.jl:242](src/Solvers/Solvers_2_CPISO.jl#L242)).

**Validation gate:** two-phase natural convection in a sealed box with phase
change off — check against a single-phase run with the same properties, and
check that a stably stratified initial condition stays stratified.

---

## Step 4 — Fixed heat flux boundary condition

**Files:** new `src/Discretise/boundary_conditions/fixedHeatFlux.jl` +
`fixedHeatFlux_interpolation.jl`, registered in `Discretise.jl`.

`FixedHeatFlux(:patch, q)` with q in W/m², positive into the domain.

- `@define_boundary FixedHeatFlux Laplacian{Linear} ScalarField` →
  `(0.0, q*area)` (pure source, no diagonal contribution), following the
  `Neumann` pattern at
  [neumann.jl:26](src/Discretise/boundary_conditions/neumann.jl#L26).
- Divergence specialisations for `Linear`/`Upwind`/`LUST`/`BoundedUpwind`,
  matching `Neumann`'s set.
- Interpolation: `T_f = T_c + q·delta/k_f` so that boundary face values are
  consistent with the imposed flux.

Note the existing generic `Neumann` is flagged in its own source as a draft
implementation for `Laplacian{Linear}` — worth confirming it is correct while
in here, since a wrong sign there would silently affect other cases.

**Validation gate:** 1D slab conduction with a known flux — recovers the
analytical linear profile and the correct wall temperature.

---

## Step 5 — Compressible ullage

> **Revised now the paper is available.** Fernandes et al. Sec. 3.2 treat the
> vapour as an **ideal gas**, with per-phase properties taken as
> temperature-dependent fits from NIST. So the full Helmholtz ullage originally
> planned here is *not* what the reference does, and §5.1 below (mandatory
> tabulation of the Helmholtz EOS) is largely obsolete for the bulk ullage:
> `rho_v = p/(R*T)` is both cheaper and closer to the reference.
>
> The Helmholtz machinery is still wanted for the **saturation** quantities that
> the phase change models need — `T_sat(p)`, `p_sat(T)` and `L(p)` — but the
> paper supplies an explicit Antoine fit for exactly those (Eq. 15,
> `A = 3.54314, B = 99.395, C = 7.726`, p_sat in bar, valid 21.01–32.27 K),
> which is a closed-form expression needing no tabulation at all.
>
> Net effect: step 5 shrinks to (a) `rho_v = p/(R*T)` per cell, (b) the
> `psi dp/dt` compressibility term in the p_rgh equation, and (c) `p_operating`.
> The allocation problem in `EOS_wrapper_H2` (§5.1) remains a real defect in that
> code path, but it is no longer on the critical path for this case.

### What was built, and two bugs found on the way

`IdealGas` EOS (`rho = p/(R*T)`, constructed from `R` or molar mass `M`), per-cell
density updates, `p_operating` as the absolute-pressure datum, and **two** terms
in the pressure equation:

- `psi * dp/dt` with `psi = sum_i alpha_i*(1/rho_i)(drho_i/dp)` — for an ideal gas
  ullage this is `(1-alpha)/p_abs`.
- `expansion = sum_i alpha_i*beta_i*DT/Dt` — the **driver**. The full low-Mach
  constraint is `div(u) + psi*Dp/Dt - sum_i alpha_i*beta_i*DT/Dt = 0`; I first
  implemented only the `psi*Dp/Dt` half, and the sealed tank then produced
  `dp/dt = 0` exactly. Heating a gas with no expansion term does nothing to the
  pressure.

Plus the pressure-work source in the energy equation, `beta*T*Dp/Dt`, which was
also missing (`S_T` had been left at zero after step 3). For an ideal gas
`beta*T = 1`, so the ullage picks up the full `Dp/Dt`; the liquid contributes
`beta*T ~ 0.33`. This is not a small correction — see the validation below.

**Corrector double-counting.** `solve_equation!` passes the solved field itself as
the time term's `prev`, so inside the PISO corrector loop the reference pressure
moved with every corrector and each one advanced the pressure by another full
`psi*dp/dt` increment. The pressure rise then scaled with `inner_loops` and grew
without bound (measured 612x the analytical value). Fixed by
`solve_pressure_compressible!`, which holds `prev` at the time-step-start
pressure so the correctors converge on one increment per step. Worth knowing that
`CPISO` has the same structure — it is not affected only because its source terms
are recomputed inside the corrector loop.

### Validation

A sealed rigid volume of ideal gas has an exact answer. Including the pressure
work turns `cp` into `cv` in the lumped balance:

    m*cp*dT/dt = Q + V*dp/dt,   dp/dt = (mR/V)*dT/dt   =>   m*cv*dT/dt = Q
    =>  dp/dt = R*Q/(V*cv),     cv = cp - R

| quantity | value |
|---|---|
| measured `dp/dt` | 510.278 Pa/s |
| exact `R*Q/(V*cv)` | 510.739 Pa/s (**0.09 % error**) |
| `R*Q/(V*cp)` (pressure work omitted) | 338.07 Pa/s |

The test asserts agreement with the `cv` form *and* disagreement with the `cp`
form, so a regression that drops the pressure-work term cannot pass silently.

**Known residual: sealed-tank mass drift.** The vapour mass should be exactly
constant (`p/T = mR/V` is invariant for a rigid closed volume). Measured drift is
2.1e-6 relative over 200 steps, and it **accumulates** — it grows with the total
pressure change, tracing to the 0.09 % rate error letting `p/T` creep. It is
currently ~0.2 % of the relative pressure change. For the full K-Site run
(O(10^6) steps, pressure roughly doubling) this needs monitoring: if the ratio
holds it implies ~0.2 % mass error at the end, which is tolerable, but it should
be measured on the real case rather than assumed.

### Original plan (retained for reference — full EOS route)

This is the largest change and the one that makes self-pressurisation possible
at all.

### 5.1 Tabulate the EOS first (mandatory, not an optimisation)

`EOS_wrapper_H2` constructs a `HelmholtzFluidConstants` containing ~20
heap-allocated `Vector`s **on every single call**
([Helmholtz_H2.jl:210](src/ModelPhysics/FluidProperties/HelmholtzEnergy/Helmholtz_H2.jl#L210)),
and the closure in `HighFidelity_Closure.jl` allocates two more per call
([HighFidelity_Closure.jl:24](src/ModelPhysics/FluidProperties/HighFidelity_Closure.jl#L24)).
It also root-finds for the density branches and returns a 2-element vector.

Calling this per cell per timestep is a non-starter: it cannot run in a
`KernelAbstractions` kernel at all (allocation + dynamic dispatch), and on CPU
it would dominate runtime for a case that needs O(10⁵) steps.

**Plan:** build lookup tables once at setup, from the existing EOS:

- `p ∈ [0.5, 5] bar`, `T ∈ [14, 45] K`, on a regular grid (resolution TBD by a
  convergence check against direct EOS evaluation).
- Tabulate ρ_v, cp_v, k_v, μ_v, ψ_v = ∂ρ/∂p|_T for the vapour branch; ρ_l, cp_l,
  k_l, μ_l, β_l for the liquid branch; and saturation curves T_sat(p),
  p_sat(T), h_fg(p).
- Store as plain `Array`s adapted to the backend; bilinear interpolation in a
  small `isbits` struct so it is GPU-safe.
- Keep the direct EOS as the reference for a unit test asserting the table
  interpolation stays within tolerance (the existing
  `test/unit_test_fluidProperties.jl` and `test/Fluids_NIST_Data/` give the
  accuracy targets to match: 0.1 % density, 2 % viscosity).

This is reusable well beyond this project.

### 5.2 Compressibility in the pressure equation

Follow the `CPISO` pattern exactly
([Solvers_2_CPISO.jl:79](src/Solvers/Solvers_2_CPISO.jl#L79)) — add a `Time`
term carrying ψ to the p_rgh equation:

```
Time{schemes.p_rgh.time}(psi_mix, p_rgh)
  - Laplacian{schemes.p.laplacian}(rDf, p_rgh)
  == -Source(divHv)
```

with `psi_mix = α ψ_l + (1−α) ψ_v`, ψ_l ≈ 0. Add `psi`/`psif` `ScalarField`s to
the `Multiphase` struct.

Consequences:
- The system is no longer singular with all-wall boundaries, so `pref` is no
  longer needed and **must not** be used (it would pin the level and suppress
  pressurisation). Worth an explicit error if `pref !== nothing` on a
  compressible multiphase run.
- `p_operating` becomes a field on `Multiphase`; absolute pressure is
  `p_abs = p_rgh + ρ·gh + p_operating`, and the EOS must be evaluated at
  `p_abs`, not at `p_rgh`.

### 5.3 Density update

`rho1_val`/`rho2_val` snapshots
([Solvers_5_Multiphase.jl:219](src/Solvers/Solvers_5_Multiphase.jl#L219)) and
the constant-argument `blend_properties!` calls must be replaced by a per-cell
EOS/table lookup each corrector. `update_mixture_properties!`
([Solvers_5_Multiphase.jl:501](src/Solvers/Solvers_5_Multiphase.jl#L501)) is the
natural home. Keep a fast path for the all-constant case so the existing
hydrostatic and mixture tests are unaffected.

### 5.4 Well-balancedness — the subtle risk

`phi_gf!` and `well_balanced_pressure_grad!`
([Solvers_5_Multiphase.jl:620](src/Solvers/Solvers_5_Multiphase.jl#L620) and
[:1246](src/Solvers/Solvers_5_Multiphase.jl#L1246)) both build the buoyancy
term from `snGrad(rho)`. Today ρ is piecewise constant, so `snGrad(rho)` is
exactly zero except at the interface, which is what makes the hydrostatic test
pass to 1e-8.

With an EOS, ρ_v varies continuously through the ullage under its own
hydrostatic gradient, so `snGrad(rho)` is non-zero everywhere and these kernels
will no longer be well balanced by construction. Expect spurious ullage currents
unless the discretisation is revisited. **This is the item most likely to cost
unplanned time in step 5** — flagging it now rather than discovering it as
"mystery convection in the ullage".

**Validation gate for step 5:** sealed tank, wall heat flux on, **phase change
off**. Pressure must rise purely by gas expansion, matching a lumped
ullage-energy-balance calculation. Separately, re-run the hydrostatic test to
confirm no regression in the constant-density path.

---

## Step 6 — The three phase change models

**Files:** new `src/ModelPhysics/2_phase_change_models.jl`;
`src/Solvers/Solvers_5_Multiphase.jl`.

### 6.1 Types and UI

Phase change slots into the existing `physics_properties` mechanism with no
change to the `Fluid{Multiphase}` constructor — it already forwards arbitrary
kwargs via `Base.structdiff` and `build_property`
([2_fluid_models.jl:330](src/ModelPhysics/2_fluid_models.jl#L330)). Adding
`phase_change = ...` costs one `build_property` method (identity — all three
types are `isbits` and need no mesh-sized storage).

```julia
abstract type AbstractPhaseChangeModel end

@kwdef struct Lee{T} <: AbstractPhaseChangeModel
    r_l::T = 0.1        # evaporation relaxation rate [1/s]
    r_v::T = 0.1        # condensation relaxation rate [1/s]
end

@kwdef struct Schrage{T} <: AbstractPhaseChangeModel
    sigma_e::T = 0.03   # evaporation accommodation coefficient [-]
    sigma_c::T = 0.03   # condensation accommodation coefficient [-]
end

@kwdef struct ModifiedEnergyJump{T} <: AbstractPhaseChangeModel
    h_int::T            # interfacial heat transfer coefficient [W/m^2/K]
end
```

### 6.2 Formulations

**Standard literature forms — to be checked against the paper (step 0.1).**
All return volumetric ṁ [kg/m³/s], positive for evaporation.

- **Lee** (no interfacial area needed):
  - T > T_sat: ṁ = r_l · α_l · ρ_l · (T − T_sat)/T_sat
  - T < T_sat: ṁ = −r_v · α_v · ρ_v · (T_sat − T)/T_sat
- **Schrage** (interfacial):
  - ṁ″ = (2σ/(2−σ)) · √(M/(2πR_u)) · ( p_v/√T_v − p_sat(T_l)/√T_l )
  - ṁ = ṁ″ · A_i
- **Modified energy jump** (interfacial):
  - ṁ = h_int · (T − T_sat) · A_i / h_fg

Interfacial area density `A_i = |∇α|`. Both `∇alpha` and
`cell_grad_magnitude!` already exist inside `MULTIPHASE`
([Solvers_5_Multiphase.jl:1109](src/Solvers/Solvers_5_Multiphase.jl#L1109)) and
can be reused as-is.

Note the paper reports Lee diverging at high coefficient values — that is a
property of the model (stiff source, explicit treatment), not a bug to fix.
Reproducing it is arguably part of reproducing the paper, but it does mean the
source term needs either sub-stepping or semi-implicit treatment to be able to
run the high-coefficient cases at all. Worth deciding deliberately rather than
discovering at run time.

### 6.3 Dispatch

```julia
phase_change_rate!(mdot, ::Nothing, args...) = nothing
phase_change_rate!(mdot, pc::Lee, alpha, T, T_sat, rho_l, rho_v, config)
phase_change_rate!(mdot, pc::Schrage, alpha, T, T_sat, p_abs, A_i, ..., config)
phase_change_rate!(mdot, pc::ModifiedEnergyJump, alpha, T, T_sat, A_i, h_fg, config)
```

### 6.4 The three sources

1. **α equation.** `advance_alpha!`
   ([Solvers_5_Multiphase.jl:476](src/Solvers/Solvers_5_Multiphase.jl#L476))
   gains `− ṁ/ρ_l` in the explicit update. It must be included **before** the
   MULES limiter computes `Qplus`/`Qminus`
   ([:1006](src/Solvers/Solvers_5_Multiphase.jl#L1006)), otherwise the source can
   push α outside [0,1] after limiting and the boundedness guarantee is lost.
2. **Pressure equation.** ∇·U = ṁ(1/ρ_v − 1/ρ_l) ≠ 0. Subtract from `divHv`
   after `div!(divHv, mdotf, config)`
   ([:377](src/Solvers/Solvers_5_Multiphase.jl#L377)).
3. **Energy equation.** `−ṁ·h_fg` source (step 3).

**Validation gate:** each model on a 1D Stefan problem (and/or the sucking
interface problem) against the analytical solution, on a plain quad mesh,
before going anywhere near the tank. These become permanent unit tests.

---

## Step 7 — Post-processing

**File:** new `src/Postprocess/Postprocess_5_multiphase.jl`.

- `ullage_average(field, alpha; threshold=0.5)` — α-weighted volume average
  over the vapour region. Needed for the paper's primary metric.
- Boil-off rate: ∫ ṁ dV over the domain.
- Interface position / liquid volume, for checking mass conservation.
- Registered through the existing `runtime_postprocessing!` mechanism so they
  can be written per write-interval like `Probe`.

---

## Step 8 — Validation matrix

3 models × 3 coefficients × K-Site fill levels (29/49/83 %) × 2 heat fluxes.
Metric: MAPE on the self-pressurisation curve. Targets from the paper's
abstract: Schrage ≤ 3.0 %, MeJ comparable when h_int is well chosen, Lee up to
11 % with divergence at high coefficients.

**Missing input:** the experimental p(t) traces are not in the repo. They need
digitising from NASA TM-103804 (a scanned document) or obtaining from the paper
authors before any of this is a validation rather than a demonstration.

---

## Runtime feasibility — flagging early

The experiment runs for O(10⁴–10⁵) s. α transport is explicit MULES, so `dt` is
Courant limited. `AdaptiveTimeStepping` exists and helps, but for a 1.1 m tank
with a resolved thermal boundary layer this is plausibly 10⁶+ steps per case,
times ~20 cases in the matrix.

Worth measuring on the wedge mesh after step 3 and before committing to steps
5–6, because if it is infeasible the answer is a semi-implicit α path or
sub-cycling, and that is better known early than late.

---

## Fixed — energy equation mixed conservative and non-conservative forms

**The single root cause of the K-Site case failing.** Found from a user
observation (velocity and temperature spiking at the phase interface, with phase
change off and gravity on).

`energy!` called `discretise!(energy_eqn, T, config)` without `rho_prev`, which
defaults to the term's own **current** flux. That gives the non-conservative time
term `rho_cp*dT/dt`, paired with a conservative divergence
`div(rho_cp_phi*T)` — leaving a spurious `T*div(rho_cp_phi)` source.

The momentum equation already avoids this by passing `rho_prev=rho_prev`
explicitly. The energy equation is far more sensitive because the error scales
with `(rho*cp)_l - (rho*cp)_v = 6.7e5`, against the momentum equation's
`rho_l - rho_v = 69.5` — a ~1e4 amplification.

**Fix:** added `rho_cp_prev` to `TwoPhaseTemperature`, snapshotted at the top of
`energy!` before the coefficients are rebuilt (and seeded in `initialise` so the
first step is valid), then passed as `rho_prev` to `discretise!`.

### Everything downstream was a symptom

| Symptom | Before | After |
|---|---|---|
| Adiabatic uniform-T drift (wedge, 5x1 ms) | 7.9 K | **4.6e-14** |
| K-Site max T deviation (200 steps) | 11.6 K | **0.0039 K** |
| K-Site max \|U\| | 14.8 m/s | **0.047 m/s** |
| Interface spike | severe | none |
| `Cg` on the pressure equation | "not positive definite" | runs |
| Adaptive time step | collapsed 0.01 -> 1e-4 | **grew** to 0.0131 |
| Courant | at limit | 0.10–0.25 |

The earlier `Cg` failure and time-step collapse were **not** separate problems, and
the well-balancedness concern from §5.4 below was not the cause either — both
hypotheses were disproved by experiment (tightening the solver tolerance to 1e-16
with DILU changed nothing; zeroing gravity did not stop the divergence).

### Why the test suite missed it

Every energy test ran on `quad40`; the wedge tests never ran the energy equation.
The "uniform temperature preserved" test had **zero velocity**, and the
compressible-ullage tests had **no interface** (`alpha = 0` everywhere), so the
coefficient jump was never exercised alongside advection.

Now covered by `unit_test_ksite_wedge_mesh.jl`: "adiabatic two-phase uniform T is
preserved" — sealed, adiabatic, gravity on, interface present, on the wedge. A
uniform field must hold exactly regardless of the flow, so it isolates advection
consistency with no source term to hide behind.

**Residual (minor):** the adiabatic static column still shows max |U| ~ 0.33 m/s
of spurious current, unchanged by this fix. That *is* plausibly the §5.4
well-balancedness item, now a second-order effect rather than a blow-up. Worth
revisiting before quantitative validation.

---

## Fixed — `reconstruct!` dropped boundary faces

**Status: FIXED.** A boundary-face pass was added; see the end of this section
for what changed and how it was verified.

`reconstruct!` sums the least-squares moment matrix over `mesh.cell_faces`,
which contains **internal faces only**. Confirmed on the generated K-Site mesh:

```
faces total    = 13390
boundary faces =  6830
internal faces =  6560
cell_faces len = 13120   ==  2 x internal      <- boundary faces absent
faces_range length: min = 2, max = 4           <- never 6
```

On a one-cell-thick wedge both wedge faces are boundary faces, so the azimuthal
direction contributes nothing and the moment matrix is **exactly rank 2 for
every cell**. `abs(det) > eps*scale^3` fails, `invdet = 0`, and the
reconstruction returns identically zero. A representative cell:

```
M = [ 4.117e-3   0.0   2.800e-4
      0.0        0.0   0.0        <- azimuthal row/column exactly zero
      2.800e-4   0.0   3.340e-3 ]
```

This silently zeroes all three call sites in `MULTIPHASE`:
`well_balanced_pressure_grad!` (momentum pressure-gradient source),
`reconstruct!(phi_g, ...)` (gravity flux), and
`reconstruct!(∇p_rghf_reconstructed, ...)` (velocity correction). The solver
runs and produces garbage rather than erroring.

It is a genuine bug, not a design choice: the call sites fill their face buffers
over *all* faces (`ndrange = length(faces)`) and the boundary entries are then
dropped, and the sibling routine `div_slip_outer!` does an explicit second pass
over boundary faces with atomics precisely because `cell_faces` excludes them.
So boundary faces are under-counted on **every** mesh, 2D and 3D — it is merely
inaccurate elsewhere and singular on a wedge.

### What was done

The general **boundary pass** was implemented (rather than the wedge-specific
rank-aware solve, which would have left the under-counting live everywhere
else). `reconstruct!` is now two passes:

1. `_reconstruct_boundary_accum!` over boundary faces, scattering the six unique
   moment-matrix components and three RHS components into per-cell scratch with
   `Atomix.@atomic` — a cell may own several boundary faces. This mirrors the
   existing `div_slip_outer_boundary_kernel!` pattern.
2. `_reconstruct_operation_2D!` / `_3D!` seed their accumulators from that
   scratch, then add the internal faces from `cell_faces` and solve.

Scratch is a `ReconstructWorkspace` allocated once in `MULTIPHASE` (9 arrays of
`n_cells`) and threaded through the three call sites; `reconstruct!` and
`well_balanced_pressure_grad!` take it as an extra argument.

Because `psif` is zero on boundary faces at every call site (those kernels form
a surface-normal gradient, and a boundary face has `ownerCells == [c, c]`),
including them imposes `u . n = 0` per boundary face — which on the wedge planes
is precisely the axisymmetry constraint.

### Verification

- **Wedge, uniform-field recovery.** With `psif_f = area_f (g . n_f)` for
  `g = (0, 0, -9.81)`, the reconstruction returns `g` to **1.1e-14** in all 3325
  cells; azimuthal component **1.4e-26**; zero cells return an identically zero
  vector. Before the fix all 3325 did. Now a permanent test in
  `test/unit_test_ksite_wedge_mesh.jl`.
- **No regression.** `2d_multiphase_hydrostatic`, `2d_multiphase_mixture` and
  `2d_multiphase_gravity` all pass (6/6).
- **Well-balancedness preserved.** The hydrostatic case was the degradation
  risk, since it has a Dirichlet `p_rgh` top where the true boundary flux is not
  zero. Measured after the fix: `mean|U| = 2.5e-10`, `max|U| = 1.3e-9`, against
  the test's `< 1e-8` gate — 40x inside it.

The scale-invariant determinant guard applied earlier to both reconstruct
kernels is a separate, unrelated fix and remains valid.

---

## Summary of files touched

| Step | New | Modified |
|---|---|---|
| 1 | `examples/0_GRIDS/ksite_wedge_5deg/` | — |
| 2 | — | `2_thermophysical_models.jl`, `2_fluid_models.jl` |
| 3 | `Energy/multiphase_energy.jl` | `energy_types.jl`, `Energy.jl`, `Solvers_5_Multiphase.jl`, `Solvers_3_solver_dispatch.jl` |
| 4 | `boundary_conditions/fixedHeatFlux*.jl` | `Discretise.jl` |
| 5 | EOS table module under `FluidProperties/` | `2_fluid_models.jl`, `Solvers_5_Multiphase.jl` |
| 6 | `2_phase_change_models.jl` | `ModelPhysics.jl`, `Solvers_5_Multiphase.jl` |
| 7 | `Postprocess_5_multiphase.jl` | `Postprocess.jl` |
