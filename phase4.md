# Phase 4 — Distributed Laplace solver: `plaplace!` (Medium)

Umbrella: `distributed_plan_detailed.md` Phase 4; first end-to-end distributed solve.

## Files
- `src/Distribute/Distribute_5_solvers.jl` — `DistributedEqn`, overloads, `plaplace!`, `prun!` skeleton.

## DistributedEqn + overloads (the §2.6 budget lands here)
```julia
struct DistributedEqn{E<:ModelEquation,S,P,H}
    eqn::E          # serial equation on the local mesh (discretise!/BCs run on this)
    solver::S       # AbstractDistributedSolver (PETScSolver)
    partition::P
    halo::H
end
```
Overloads on existing generics, dispatched on `DistributedEqn`:
- `solve_system!(deqn, setup, result, component, config)` → `passemble!` + `psolve!` +
  copy owned x back into `phi.values` + `sync!(phi)`.
- `residual(deqn, component, config)` — `R`/`Fx` restricted to rows `1:n_owned`;
  `MPI.Allreduce` of `sum(R)` and `norm(b)²`; every rank returns the identical global value.
- `setReference!(deqn, pRef, cellID, config)` — nzval/b edit applied ONLY on the rank
  owning global `cellID` (lookup via partition block ranges).
- `solve_equation!(deqn, phi, BCs, setup, config)` — same orchestration as serial
  (discretise! → BCs → relaxation on the inner `eqn`), distributed solve + residual.
  Preconditioner update (`update_preconditioner!`) skipped — PETSc PC owns preconditioning.

## plaplace! and prun!
- `plaplace!` mirrors serial `laplace!`: build eqn on local mesh → wrap as
  `DistributedEqn` → iterate discretise/BC/solve → per-rank output.
- `prun!(model, config; petsc_options="", kwargs...)` dispatch skeleton mirroring
  `run!` in `Solvers_3_solver_dispatch.jl` (keys on model); errors with an instructive
  message when no solver extension is loaded or `model.domain` is not a `DistributedMesh`.
- Per-rank VTK: rank-suffixed files + rank-0 `.pvtu` master (minimal; polish Phase 8).

## Tests (`test/distributed/test_laplace.jl`, n = 1, 2, 4, 8)
- 3D box diffusion: per-cell solution vs serial `laplace!` gathered via `orig_cells`,
  relative L2 < 1e-8; identical field across rank counts.
- Residual histories match serial to tolerance.
- Reference-cell pinning: pressure-level consistency across rank counts.

## Exit criteria
Identical converged fields for n=1,2,4,8; n=1 matches serial `laplace!` exactly
(within solver tolerance). `.pvtu` opens in ParaView with correct global field.
