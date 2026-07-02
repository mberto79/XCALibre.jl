# Phase 0 — Scaffolding & environment (Low)

Umbrella: `distributed_plan_detailed.md` (§1 locked decisions, §2 architecture). Some items
are folded into Phase 1 (module skeleton, MPI/Metis deps, compat bump) so Phase 1 is runnable.

## Deliverables
1. `src/Distribute/Distribute.jl` module skeleton, included + reexported from `src/XCALibre.jl`. [done in Phase 1]
2. Project.toml: `MPI`, `Metis` in `[deps]`; `julia = "1.10"` compat. [done in Phase 1]
   `MPIPreferences` added only when cluster/system-binary work starts.
3. `[weakdeps]` `PETSc` → `[extensions]` `XCALibrePETScExt`; empty `ext/XCALibrePETScExt.jl`
   defining `PETScSolver <: Distribute.AbstractDistributedSolver` (interface stub lives in
   `Distribute_0_types.jl`).
4. MPI test harness: `test/distributed/runtests_mpi.jl` that shells out
   `$(MPI.mpiexec()) -n N julia --project <testfile>`; precompile serially first
   (MPI precompile race). Single-process tests (Phase 1) do NOT need this harness.
5. CI: GitHub Actions job, CPU only, runs single-process distributed tests + `-n 2`/`-n 4`
   MPI tests as later phases add them.

## Exit criteria
- `using XCALibre` unchanged for serial users (no MPI.Init required; Windows serial OK).
- `using XCALibre, PETSc` activates the extension (verified by `Base.get_extension`).
- `mpiexec -n 2` smoke test passes: each rank prints rank/nranks via a trivial helper.
