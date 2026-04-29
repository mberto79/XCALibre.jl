# EFM (Thin Film) Solver — Potential Issues

Review of `src/Solvers/Solvers_1_FilmModel.jl` (branch `HM/ETFM`), driven by symptoms in `efm_test.jl` (Case 8, ϕ=90, σ=0.069, β=3, inner_loops=10): U and h decouple / numerical artefacts appear around iteration 800–1000.

Architecture must remain unchanged. Fixes are ordered by likelihood of being the root cause; apply one at a time and rerun the test case to discriminate.

## Orientation

- Solver entry: `filmModel!` → `setup_FilmModel_Solver` → `FilmModel`.
- U_eqn: `Time(hf, U) + Divergence(phif, U) + Si(nu_h, U) == -Source(h∇PL) + Source(Ph) + Source(τθw)`
- h_eqn: `Time(h) - Laplacian(Df, h) == -Source(divPhi) + Source(Sm)`
- PISO-like inner loop: predictor U solve → `remove_film_pressure_source!` strips pressure from `b` → inner loop iterates (`H!`, mdotf, divPhi, h solve, recompute capillary/hydrostatic, `correct_film_velocity!`, `correct_mass_flux2!`).
- `hf` is updated as a side-effect of `laplacian!(Δh, hf, h, ...)` (Calculate_0_laplacian.jl:8 calls `interpolate!(phif_in, phi_in, config)` then `correct_boundaries!`). It is not interpolated explicitly anywhere in this solver.
- `Df = -rDf * hf * dot(n, G) * area` (gravity only — no σ contribution).
- At ϕ=90 (the active test case): `n=[1,0,0]`, `G=[0,0,-9.8]`, `dot(n,G)=0` ⇒ `Df=0`, `P_hydrf=0`. Surface tension is the only film pressure; it enters fully explicitly via `correct_film_velocity!` and the RHS source.

## Fix 1 — `Time(hf, U)` reads face values at cell indices (DEFINITE BUG)

**Location**: `Solvers_1_FilmModel.jl:49`

```julia
Time{schemes.U.time}(hf, U)
```

`hf` is a `FaceScalarField`. The Euler time scheme (`Discretise_1_schemes.jl:32–42`) does:

```julia
vol_rdt = term.flux[cID]*volume/runtime.dt[1]
ac = vol_rdt
b  = prev[cID]*vol_rdt
```

`cID` is a **cell** index (1..N_cells); `term.flux` is `hf` whose `.values` is sized by **N_faces**. `hf.values[cID]` therefore reads the value of *some unrelated face* (the face whose global index happens to equal cID — typically a boundary face for small cID, an internal face for larger cID). Result: the U_eqn diagonal contribution from the time term is wrong at every cell.

**Why every other call site is fine**: search shows all other `Time{...}` usages pass a cell-centered scalar:
- `Time{...}(rho, U)` in CPISO/CSIMPLE/SIMPLE
- `Time{...}(rho, k)`, `Time{...}(rho, omega)` in turbulence models
- `Time{...}(rhocp, T)` in LAPLACE
- `Time{...}(psi, p)` in CPISO

`hf` is the only site passing a face field.

**Why the simulation runs at all**: `hf[cID]` returns positive, bounded values of the right order of magnitude (some face's h, between `h_floor=1e-15` and `h_inlet=0.05`). Diagonal stays positive, so the linear solve still converges — just with the wrong coefficient. Errors accumulate as h grows over the simulation, consistent with stable behavior for ~hundreds of iterations followed by drift.

**Propagation**: the bad diagonal flows through `inverse_diagonal!` → `rD` → `Hv = (b - sum(an*U_n))/D` → `correct_film_velocity!`. So the entire PISO correction is built on a corrupted A.

**Fix**: change `hf` → `h` (the cell-centered `ScalarField` already destructured at line 25 / 92).

```julia
Time{schemes.U.time}(h, U)
```

This is the only fix to apply first. Rerun the test case. If decoupling moves later or disappears, this was the cause. If it still hits at ~800–1000 iters, the explicit surface-tension stiffness (architectural item below) is the actual driver.

## Fix 2 — `cell_nsign[i]` mis-indexed in `_correct_mass_flux2!` (DEFINITE BUG, currently silent)

**Location**: `Solvers_1_FilmModel.jl:563–572`

```julia
@kernel function _correct_mass_flux2!(mdotf, h, Df, mesh)
    i = @index(Global)
    ownerCells = mesh.faces[i].ownerCells
    snGrad = mesh.cell_nsign[i]*(h[ownerCells[2]]-h[ownerCells[1]])/mesh.faces[i].delta
    ...
    mdotf[i] -= Df[i] * len_me * snGrad
end
```

`cell_nsign` is a per-cell-faces flat array, sized by `Σ |cell.faces_range|`. The discretise kernel uses it as `cell_nsign[fi]` where `fi` is a **per-cell** index inside `faces_range` (`Discretise_2_generated_distretisation.jl:59,145`). Indexing it with global face id `i` returns a random ±1.

Reference: the standard `_correct_mass_flux!` in `Solvers_1_SIMPLE.jl:369–386` does not use cell_nsign at all — it pulls the off-diagonal coefficient directly via `spindex(rowptr, colval, cID1, cID2)`.

**Currently masked**: at ϕ=90, `Df=0`, so the entire correction is multiplied by zero. Will activate as soon as ϕ ≠ 90 (e.g., Case 1, ϕ=5).

**Fix candidates** (apply if Fix 1 doesn't resolve the issue, or before testing other ϕ values):
- Drop `cell_nsign[i]` — `(h[ownerCells[2]] - h[ownerCells[1]])/delta` is already a signed normal gradient if ownerCells[1] is owner and the face normal is conventionally oriented from owner to neighbour. Verify the convention before committing.
- Or follow the SIMPLE pattern: skip boundary faces (loop only `n_bfaces+1:n_faces`), use `spindex` to pull the relevant matrix coefficient.

## Fix 3 — Inconsistent Ph initialisation (MINOR)

**Location**: `Solvers_1_FilmModel.jl:227` vs `:321`

- Init (line 227): `Ph_local = (g*sind(coeffs.ϕ)*h[i])/rho.values[1] .*plate_tangent_vector`
- Loop (line 321): `Ph_local = (g*sind(coeffs.ϕ)*h[i]) .*plate_tangent_vector`

The loop form is consistent with the equation: U_eqn LHS has `Time(h, U)` and `Divergence(phif, U)` (no ρ factor — already in d(hu)/dt form, depth-integrated). RHS sources should also be in those units. The init `/rho` is a leftover and only affects iteration 1.

**Fix**: drop `/rho.values[1]` from line 227.

## Architectural note — NOT a code bug

At ϕ=90, `Df=0`, so:
- h_eqn loses its implicit pressure-correction Laplacian entirely.
- `correct_mass_flux2!` is a no-op (Df=0).
- Surface tension enters only explicitly: as `-Source(h∇PL)` on U_eqn's RHS, removed from b before `H!`, then added back in `correct_film_velocity!` via `(∇P_hydr + ∇P_surf)*h*rD/ρ`.

This is the algorithm as written, not a bug. But for σ-driven cases (Case 8 has σ=0.069 with β=3 contact-line forcing), fully explicit capillary stiffness is the structural reason small errors compound over many iterations. If the three bug fixes above don't eliminate the artefacts, the model needs a σ contribution in `Df` (e.g., `Df_total = -rDf*hf*(g_n + σ*(...))*area`), which is a model upgrade outside "minimal changes."

## Minor cleanup (skip unless touching the file)

- `mu = nu.values*rho.values` at line 114 — computed, unused.
- `n_cells = length(mesh.cells)` at line 178 — computed, unused.

## Test methodology

`efm_test.jl` Case 8 currently selected (ϕ=90, β=3, σ=0.069, inlet_flow_rate=25.53e-5, h_inlet=0.05, h_floor=1e-15, Δt=1e-4, 2000 iters, inner_loops=10, write_interval=100, adaptive Co=0.01). Apply fixes one at a time and rerun. Decoupling reportedly appears between iter ~800–1000.

Note from CLAUDE.md: do not run the simulation under foreground Bash (Julia JIT + warmup exceeds 2-minute timeout). Use `run_in_background=true` or run via the user's Julia REPL session.

# Latest Implementation: capillary timestep ceiling

Implemented a capillary timestep ceiling for the EFM model.

  Changed:

  - src/ModelPhysics/0_type_definition.jl:147: added
    capillary_dt, defaulting to 1e-6.
  - src/Solvers/Solvers_1_FilmModel.jl:246: clamps runtime.dt
    before each iteration.
  - src/Solvers/Solvers_1_FilmModel.jl:349: clamps again after
    the normal Courant adaptive update, so adaptive growth cannot
    escape the capillary limit.
  - src/Solvers/Solvers_1_FilmModel.jl:524: added
    limit_capillary_dt!.
    - Fixed capillary timestep reporting/use so dt_cpu is copied
    after applying the cap and after adaptive updates: src/
    Solvers/Solvers_1_FilmModel.jl:101

  Result:

  - Reproduced the failure with the original adaptive 1e-4 path:
    it goes NaN around time ≈ 0.054, with dt having grown above
    the capillary-stable range.
  - With capillary_dt=1e-6, Case 8 completed 2000 iterations
    without NaN.

  Caveat: this is a stability workaround, not the final physical
  model. At 1e-6, 2000 iterations only reaches time ≈ 0.002, so
  it confirms explicit capillary stiffness but does not replace a
  proper implicit fourth-order capillary treatment in the h
  correction.