# LH2 tank self-pressurisation in XCALibre — gap analysis

Target: reproduce the NASA K-Site liquid hydrogen tank self-pressurisation cases
from *"A CFD comparison of interfacial phase change models for boil-off,
self-pressurisation and thermal stratification in liquid hydrogen storage
tanks"*, Int. J. Heat Mass Transfer **256** (2026) 128067, with all three
interfacial phase change models (Schrage, Modified energy jump, Lee) selectable
from the user interface.

Case skeleton: `test/0_TEST_CASES/3d_LH2_ksite_selfpressurisation.jl`.

---

## 0. What the reference case needs

| Quantity | Value | Source |
|---|---|---|
| Tank shape | ellipsoid, major:minor axis ratio 1.2 | NASA TM-103804 |
| Major diameter | 2.2 m | NASA TM-103804 |
| Volume | 4.89 m³ | NASA TM-103804 |
| Fill levels | 29 %, 49 %, 83 % by volume | NASA TM-103804 |
| Wall heat flux | 2.0 and 3.5 W/m² | NASA TM-103804 |
| Gravity | 1 g, normal gravity | NASA TM-103804 |
| Phase change models | Schrage, Modified energy jump (MeJ), Lee | paper |
| Reported accuracy | Schrage ≤ 3.0 % MAPE; MeJ comparable; Lee up to 11 %, diverges at high coefficient | paper abstract |

**Caveat:** the paper full text is paywalled (Elsevier and ResearchGate both
returned 403). Everything above beyond the model names comes from the open
literature on K-Site and from the paper abstract. The exact model formulations,
the three coefficient values swept per model, and the initial pressure /
temperature used in the paper still need to be read off the PDF. Items marked
`# TO CONFIRM` in the case file are the ones this affects.

---

## 1. Blockers — the case cannot run at all without these

### A. No energy equation in the multiphase solver
`MULTIPHASE` ([Solvers_5_Multiphase.jl:168](src/Solvers/Solvers_5_Multiphase.jl#L168))
advances `alpha`, `U` and `p_rgh` only. There is no temperature, no enthalpy,
no `energy!` call.

Worse, it fails *silently*: `run!` dispatches on `F<:Multiphase` at
[Solvers_3_solver_dispatch.jl:48](src/Solvers/Solvers_3_solver_dispatch.jl#L48)
regardless of the energy parameter, so `Energy{SensibleEnthalpy}(...)` is
accepted by the constructor and then ignored. `SensibleEnthalpy` would not work
anyway — it reads `model.fluid.cp`, `model.fluid.R` and `model.fluid.gamma`
([he_energy.jl:239](src/ModelPhysics/Energy/he_energy.jl#L239) onwards), none of
which exist on `Multiphase`.

**Needed:** a two-phase enthalpy (or temperature) equation using α-blended
`rho*cp` and `k`, with the phase change latent heat source.

### B. Both phases are hard-wired incompressible
The solver snapshots scalar densities once:

```julia
rho1_val = phases[main].rho[1]
rho2_val = phases[secondary].rho[1]
```
([Solvers_5_Multiphase.jl:219](src/Solvers/Solvers_5_Multiphase.jl#L219))

and blends them as constants through `blend_properties!`. Even if
`Phase(rho=HelmholtzEnergy(H2()))` were passed, `build_phase` allocates a
`ScalarField` but the solver only ever reads element `[1]`.

**This is the single biggest blocker.** Self-pressurisation *is* the ullage
compressibility response — with a constant vapour density the tank pressure
cannot rise no matter how good the phase change model is.

**Needed:** ρ_v = ρ_v(p, T) evaluated per cell each step, plus the
compressibility term ψ ∂p/∂t in the pressure equation.

### C. No thermodynamic pressure level for a sealed domain
The pressure equation
([Solvers_5_Multiphase.jl:138](src/Solvers/Solvers_5_Multiphase.jl#L138)) is a
pure Laplacian of `p_rgh` with `div(Hv)` as source — singular when every
boundary is a wall. The existing escape hatch is the `pref` kwarg, which pins
the level, i.e. it *prevents* pressurisation.

There is also no concept of operating/absolute pressure. Note
`operating_pressure = 0.0` at
[2d_multiphase_hydrostatic.jl:42](test/0_TEST_CASES/2d_multiphase_hydrostatic.jl#L42)
is a dead variable — never used.

**Needed:** `p_operating` on the fluid model, and a compressible p_rgh equation
so the pressure level is determined by ullage mass and volume rather than a BC.

### D. The three phase change models do not exist
No `Lee`, `Schrage` or energy-jump code anywhere in `src/`. The only trace of
intent is a comment in
[HighFidelity_Closure.jl:5](src/ModelPhysics/FluidProperties/HighFidelity_Closure.jl#L5):
*"Lee model is required to bring it back to physical state."*

### E. No mass source anywhere in the transport
Even with `mdot` computed, three places need a source that is not there:

- `advance_alpha!` ([Solvers_5_Multiphase.jl:476](src/Solvers/Solvers_5_Multiphase.jl#L476))
  updates α with no source term.
- The pressure equation assumes ∇·U = 0. With phase change
  ∇·U = ṁ(1/ρ_v − 1/ρ_l) ≠ 0.
- The (nonexistent) energy equation needs −ṁ·h_fg.

The MULES limiter bounds will also need revisiting once α has a source.

### F. No saturation coupling
`EOS_wrapper_H2` already returns `T_sat` and `latentHeat`
([HighFidelity_Closure.jl:36](src/ModelPhysics/FluidProperties/HighFidelity_Closure.jl#L36)),
but nothing in the multiphase path calls it. All three models need a per-cell
`T_sat(p)`; Schrage additionally needs `p_sat(T)`.

### G. No mesh
The K-Site tank is an ellipsoid. XCALibre has no axisymmetric/wedge treatment,
so this must be a thin 3D wedge with `Symmetry` on the wedge planes (equivalent
to an axisymmetric discretisation) or a full 3D mesh. Neither exists in
`examples/0_GRIDS/`. Note that gmsh reading is **not** wired into the package —
`src/Mesh/gmsh/` is not included by `src/XCALibre.jl`. Available readers are
`UNV2D_mesh`, `UNV3D_mesh` and `FOAM3D_mesh`.

---

## 2. Required, but smaller

### H. Phase thermal properties are allocated and then ignored
`PhaseState` carries `k`, `cp` and `beta`
([2_fluid_models.jl:229](src/ModelPhysics/2_fluid_models.jl#L229)) and
`build_phase` allocates them — but nothing in the codebase ever writes or reads
them (grep confirms zero uses outside the struct definition). `Phase(; rho, mu)`
does not even accept them as keywords.

### I. No fixed-heat-flux boundary condition for energy
K-Site is specified by wall heat flux (2.0 / 3.5 W/m²). The only energy BC is
`FixedTemperature` (+ `Enthalpy`/`IEnergy` variants). The generic `Neumann` BC
takes a gradient value but its `Laplacian{Linear}` specialisation is flagged in
the source as a draft
([neumann.jl:29](src/Discretise/boundary_conditions/neumann.jl#L29)).

### J. Surface tension is a bare constant
`VOF(sigma=...)` takes a `Float64`. `calculate_surface_tension(H2(), T)` exists
([surface_tension.jl:27](src/ModelPhysics/FluidProperties/surface_tension.jl#L27))
but is not reachable from the VOF model. Probably acceptable to leave constant
for LH2 near 20 K (σ ≈ 1.9e-3 N/m) — worth noting, not a blocker.

### K. Turbulence / buoyancy
Rayleigh number in the K-Site tank is ~10¹⁰. XCALibre's k-ω / k-ω SST have no
buoyancy production term. The usual choices are laminar (common for these
validations) or SST with a buoyancy source. Recommend starting laminar and
flagging it as a modelling assumption.

### L. Run length vs. explicit α transport
The experiment runs for O(10⁴–10⁵) s. α transport is explicit MULES, so `dt` is
Courant limited. `AdaptiveTimeStepping` exists and helps, but this is likely to
be the practical bottleneck — worth a look at a semi-implicit α path or
sub-cycling before committing to the full validation matrix.

### M. Post-processing
No ullage-averaged pressure, boil-off rate, or interface-position monitors. The
paper's metric is MAPE on the pressurisation curve, so at minimum an
ullage-average reduction (α-weighted) is needed. `Probe` exists and could be
extended.

---

## 3. Proposed user interface

Phase change is a property of the two-phase *fluid*, so it slots into the
existing `physics_properties` mechanism with no change to the `Fluid{Multiphase}`
constructor: `Fluid{Multiphase}` already forwards arbitrary kwargs through
`Base.structdiff` and `build_property`
([2_fluid_models.jl:330](src/ModelPhysics/2_fluid_models.jl#L330)). Adding
`phase_change = ...` therefore costs one `build_property` method.

```julia
abstract type AbstractPhaseChangeModel end

"""Lee (1980) relaxation model. Coefficients are relaxation rates [1/s]."""
@kwdef struct Lee{T} <: AbstractPhaseChangeModel
    r_l::T = 0.1      # evaporation
    r_v::T = 0.1      # condensation
end

"""Hertz-Knudsen-Schrage. Coefficients are accommodation coefficients [-]."""
@kwdef struct Schrage{T} <: AbstractPhaseChangeModel
    sigma_e::T = 0.03
    sigma_c::T = 0.03
end

"""Modified energy jump. Coefficient is an interfacial HTC [W/m^2/K]."""
@kwdef struct ModifiedEnergyJump{T} <: AbstractPhaseChangeModel
    h_int::T
end
```

Used as:

```julia
fluid = Fluid{Multiphase}(
    model  = VOF(cAlpha=1.0, sigma=1.93e-3),
    phases = (Phase(rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0, beta=0.0164),
              Phase(rho=HelmholtzEnergy(H2()), mu=1.11e-6, k=0.0169, cp=12200.0)),
    phase_change = Schrage(sigma_e=0.03, sigma_c=0.03),
    p_operating  = 117.2e3,
    gravity      = gravity
)
```

All three dispatch on a single internal entry point returning a volumetric mass
transfer rate `mdot` [kg/m³/s], positive for evaporation:

```julia
phase_change_rate!(mdot, ::Lee,                alpha, T, T_sat, rho, phases, config)
phase_change_rate!(mdot, ::Schrage,            alpha, T, T_sat, p, A_i, ..., config)
phase_change_rate!(mdot, ::ModifiedEnergyJump, alpha, T, T_sat, A_i, h_fg, config)
phase_change_rate!(mdot, ::Nothing, args...) = nothing   # no phase change
```

Schrage and MeJ need the interfacial area density A_i = |∇α|. Both `∇alpha` and
`cell_grad_magnitude!` already exist inside `MULTIPHASE`
([Solvers_5_Multiphase.jl:1109](src/Solvers/Solvers_5_Multiphase.jl#L1109)) and
can be reused directly.

---

## 4. Suggested order of work

1. **Mesh** — K-Site wedge (blockMesh → `FOAM3D_mesh`), verify `Symmetry` works
   on the non-axis-aligned wedge planes. Independent of everything else.
2. **Phase thermal properties** — make `Phase` accept `k`, `cp`, `beta` and
   actually populate `PhaseState`. Small, self-contained.
3. **Two-phase energy equation** in `MULTIPHASE`, isothermal-ullage first
   (no phase change), validated against a pure natural-convection case.
4. **Fixed-heat-flux BC** for the energy field.
5. **Compressible ullage** — ρ_v(p,T) via the Helmholtz EOS + ψ ∂p/∂t in the
   pressure equation + `p_operating`. Validate with a sealed-tank heat-soak
   with phase change off (pressure rises purely by gas expansion).
6. **Phase change models** — types, `phase_change_rate!`, and the α / pressure /
   energy sources. Verify each on a 1D Stefan problem before the tank.
7. **Post-processing** — ullage-average pressure, boil-off rate.
8. **Validation matrix** — 3 models × 3 coefficients × K-Site fill levels.

Steps 3, 5 and 6 are substantial changes to `Solvers_5_Multiphase.jl` and
`2_fluid_models.jl`.
