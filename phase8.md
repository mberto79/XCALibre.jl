# Phase 8 — HYPRE, AD (optional add-on), offline partitioning, docs (Medium/Low)

Reprioritised 2026-07-03: AD/adjoint (formerly Phase 7) moved here as an optional
add-on; usability items (GPU-native validation, I/O, F32) moved up to Phase 7.
Independent workstreams; order by demand.

## 1. HYPRE extension
- `[weakdeps]` `HYPRE` → `ext/XCALibreHYPREExt.jl`; `HYPRESolver <: AbstractDistributedSolver`
  via the IJ assembler (`start_assemble!`/`assemble!`/`finish_assemble!`); BoomerAMG default
  for the pressure Poisson.
- Backend selection: `prun!(...; linear_backend=:petsc|:hypre)`; `SolverSetup` untouched.
- Adjoint caveat: no first-class transpose solve — SPD pressure Poisson is self-adjoint;
  error for non-symmetric adjoint requests through HYPRE.

## 2. AD / adjoint boundary (optional add-on)
Umbrella: `distributed_plan_detailed.md` Phase 7; `distributed_plan.md` §7 (AD strategy).
Local kernels stay AD-differentiable; MPI + linear solve get custom rules. No
differentiation through PETSc internals.
- `ChainRulesCore.rrule` for `halo_exchange!`: pullback = `halo_exchange_adjoint!`
  (ghost cotangents summed into owning cells via `unpack_add!`; kernels exist since
  Phase 2). Copy-semantics wrapper as in `distributed_plan.md` §5.
- `psolve_transpose!` in the PETSc extension via `KSPSolveTranspose`; `rrule` for the
  distributed solve: `b̄ = solve(Aᵀ, x̄)`; `Ā = -b̄ ⊗ x` restricted to the local sparsity
  (lazy — only materialise entries present in the CSR pattern).
- `rrule`s for `pnorm`/`pdot`/`pmean`: pullback broadcasts the (identical-on-all-ranks)
  seed to local contributions — no communication in the pullback.
- ChainRulesCore becomes a dep of `Distribute` (tiny, no weight concern).
- Tests (`test/distributed/test_adjoint.jl`, n=2,4): adjoint identity
  `⟨v̄, Hx⟩ == ⟨Hᵀv̄, x⟩` to 1e-12; gradient of a scalar loss on a small distributed
  diffusion case vs serial reverse-mode AD, 1e-6, rank-count independent; FD spot-check.
- Risk: Enzyme/KA version pinning (Julia 1.11 `setindex!` regressions) — pin tested
  versions; hand-written `rrule` fallback for any kernel that fails.

## 3. Offline partitioning
- `partition_mesh(meshfile, nparts; dir)`: runs Phase 1 pipeline on one node, writes
  per-rank local meshes + Partition + procs as JLD2; `distribute(dir; comm)` loads in
  parallel (each rank reads its own file — no rank-0 memory bottleneck).
- Test: offline vs online produce identical `DistributedMesh` (hash the arrays) and
  identical solutions.

## 4. Docs
- Docs pages: workflow (§2.2 umbrella), cluster setup (MPIPreferences system binary,
  CUDA-aware env — reuse `dev/gpu_native_setup.md` content), CI notes, limitations
  (laminar-only v1, no cross-partition periodics, Windows serial-only, 1-layer halo ⇒
  no least-squares gradients).

## Exit criteria
HYPRE pressure solve matches PETSc within tolerance; AD identities/gradients green at
n=2,4 (if built); offline == online; docs build (`julia --project=docs docs/make.jl`).
