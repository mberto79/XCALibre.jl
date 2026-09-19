# Plan review: distributed (MPI) polish for production
> Tick [x] the options you want, add notes inline, then tell Claude to continue.

Context findings (already verified in the code):
- `run!(model, config)` ALREADY dispatches to the distributed path automatically when
  `model.domain` is a DistributedMesh (via `is_distributed_mesh`/sync!/wrap_eqn seams).
  A `prun!` twin is NOT needed — it would duplicate a working dispatch.
- Solver config to PETSc DOES partly work: `Cg/Bicgstab/Gmres` → KSP type, `Jacobi/BoomerAMG`
  → PC type, and `rtol/atol/itmax` reach PETSc via `KSPSetTolerances`. So atol/rtol in
  `2D_cylinder_U_mpi.jl` ARE configuring PETSc.
- BUT `convergence` (the serial residual target) is silently IGNORED in distributed, and there
  is no visible confirmation of what PETSc actually received — hence the "am I even configuring
  it?" doubt. That is a discoverability/honesty gap, not a wiring gap.

## Hidden assumptions (CONFIRMED)
- [x] Serial behaviour and API must stay byte-for-byte unchanged; all work is behind existing seams.
- [x] petsc_options string stays the escape hatch for any non-curated KSP/PC/HYPRE option.
- [x] "Production ready" for THIS round = ergonomics + config clarity + clean output + docs
      + KOmegaSST; full multi-GPU scaling validation, CUDA-aware MPI, AMD parity and AD NOT in
      scope (need lab hardware).

## Decisions needed

### D1: Launch boilerplate (#2) — CHOSEN: A
- [x] A: keep `run!` as the single entry; add a `distribute(reader::Function; comm)` overload
      doing the root-only read. Script drops to ~2 distributed lines. No prun!.
- [ ] B: add a `prun!` wrapper.
- [ ] C: document only.

### D2: `convergence` in distributed (#4) — CHOSEN: A
- [x] A: keep atol/rtol/itmax as PETSc knobs; if atol=rtol=0, fall back to convergence→PETSc
      atol; always echo resolved KSP/PC/atol/rtol/itmax once on rank 0.
- [ ] B: echo only, no map.
- [ ] C: warn if ambiguous.

### D3: BoomerAMG / HYPRE tuning surface (#4) — CHOSEN: B
- [ ] A: petsc_options passthrough + docs only.
- [x] B: add curated `BoomerAMG(; strong_threshold=…, …)` expanding to petsc options
      (passthrough kept as escape hatch).

### D4: Rank-aware logging (#3) — CHOSEN: A (default)
- [x] A: non-root ranks get a Warn+-only global logger at init/distribute; keeps @warn/@error.
- [ ] B: leave as-is.

### D5: Distributed turbulence (#1 pending) — CHOSEN: B
- [ ] A: defer turbulence.
- [x] B: include KOmegaSST as Phase 5 this round. LKE/LES still deferred.

## Out of scope (CONFIRMED)
- [x] Multi-GPU scaling validation, CUDA-aware MPI variants (need lab GPUs).
- [x] AMD ext parity verification (mirrored, lab-unverified).
- [x] Phase 8 §4 automatic differentiation add-on.
