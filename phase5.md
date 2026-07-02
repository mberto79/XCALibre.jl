# Phase 5 — Incompressible SIMPLE/PISO: `psimple!`, `ppiso!` (High)

Umbrella: `distributed_plan_detailed.md` Phase 5. `psimple!` is a contained copy of
`SIMPLE` (`src/Solvers/Solvers_1_SIMPLE.jl`) living in `Distribute` — serial files
untouched (module-isolation constraint). Maintenance rule: any upstream change to
`SIMPLE`/`PISO` must be mirrored; revisit a shared-hook refactor post-v1.

## Setup (`psetup_incompressible_solvers`)
Mirror `setup_incompressible_solvers`: same field/equation construction on the local mesh,
then wrap `U_eqn`, `p_eqn` as `DistributedEqn` (Phase 4). Skip serial preconditioner
allocation (PETSc PC). Turbulence: `RANS{Laminar}` only in v1 — hard error otherwise.
Periodic BCs that Metis cut across ranks: detected and rejected at `distribute` time.

## Sync map (each = one `sync!`; derived from the actual SIMPLE body)
1. Start: sync U, p before initial `interpolate!`/`grad!`; sync ∇p after initial `grad!`.
2. After momentum `solve_equation!` → sync U (already done inside distributed
   solve_equation!; `H!` reads neighbour U through off-diagonals).
3. After `inverse_diagonal!` → sync rD (ghost diagonals are garbage) before
   `interpolate!(rDf, rD, config)`.
4. After `H!(Hv, ...)` → sync Hv before `interpolate!(Uf, Hv, config)`.
5. After pressure solve + `explicit_relaxation!` → sync p (relaxation changes owned p;
   ghosts must see the relaxed values).
6. After `grad!(∇p, ...)` → sync ∇p (3-wide) — consumed by `limit_gradient!`,
   `nonorthogonal_face_correction`, gradient face interpolation.
7. Non-orthogonal corrector loop: repeat 5–6 per corrector.
8. No sync needed: `correct_mass_flux!` (p synced; `aN = A[owned, ghost]` is in the local
   CSR), `correct_velocity!` (per-owned-cell), `flux!`/`div!` (inputs already synced).
9. Convergence: residuals are already global (Phase 4) — every rank takes the same branch;
   no extra Allreduce. `mean`/`norm` inside relaxation/limiters over cells: audit each —
   any global reduction goes through `pnorm`/`pmean`.

## ppiso!
After `psimple!` validates: same sync map inside the PISO inner loops + transient stepping;
`time_step!`/CFL checks use global reductions.

## Tests (n = 1, 2, 4, 8, CPU; cases from existing verification suite)
- Lid-driven cavity + 2D backward-facing step: fields match serial within 1e-6 relative;
  centreline profiles + integral quantities (reattachment length) match; solution
  independent of rank count.
- Residual histories: same order/monotonicity as serial (Krylov-vs-KSP arithmetic only).
- `ppiso!`: transient cavity spin-up matches serial time history.

## Exit criteria
Both cases green at all rank counts; n=1 distributed == serial baseline. Record
iteration counts + wall times in dev/STATE as the scaling baseline for Phase 6.
