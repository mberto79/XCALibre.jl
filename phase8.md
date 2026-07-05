# Phase 8 — solver unification, offline partitioning, HYPRE, AD, docs

Reprioritised 2026-07-04 (user decision): solver/distributed unification is TOP priority —
the p-twin per solver pattern is a maintenance dead end. Full analysis: dev/arch_review.md.

Cross-cutting rule (applies to every GPU-touching item, all phases): any CUDA-ext feature
must be mirrored in ext/XCALibre_AMDExt.jl (and kept API-symmetric: KernelAdaptor adapt,
bind_device!, petsc_device_info). AMD is not lab-testable locally — mirror + parse-check,
flag as lab-unverified.

## 1. Solver unification (TOP — dev/arch_review.md is the spec)
Goal: ONE solver body runs serial and distributed; delete plaplace!/psimple!/ppiso!;
parallelism lives below the solver-author API (OpenFOAM model). Key lever: run! already
dispatches on Physics domain D, so D<:DistributedMesh routes to the same simple!/piso!.
Seams (serial method = inlined no-op, zero serial cost — hard constraint):
- S1 sync!: self-syncing tails on grad!/inverse_diagonal!/H!/correct_velocity!/
  explicit_relaxation!; DistributedMesh method = halo_exchange! with width-keyed cache on dm.
- S2 wrap_eqn factory in setup: identity (serial) / DistributedEqn (kept as the below-API
  layer with its solve/residual/setReference overrides + PETScSolver).
- S3 residual/setReference!: already dispatch on DistributedEqn — keep.
- S4 fold min/max canonical-row into serial _make_symmetric!/_correct_mass_flux!
  (correct+cheap serially) — deletes the p-variants.
- S5 global_max seam in max_courant_number!. S6 writer dispatch on mesh type.
Migration A–E with gates (serial suite + perf budgets FLAT, distributed gates green):
see arch_review.md §4. Phase E extends wrap_eqn to turbulence/energy eqns — every future
model becomes distributed-capable for free (the whole point).
Follow-on once unified: cross-partition periodic BCs (partition graph must include periodic
adjacency; halo maps periodic ghosts). Today prun! hard-rejects periodic —
examples/3D_cascade_mpi_GPU.jl uses Symmetry as a stand-in.

## 2. Offline partitioning
- partition_mesh(meshfile, nparts; dir): Phase 1 pipeline on one node → per-rank local
  meshes + Partition + procs as JLD2; distribute(dir; comm) loads in parallel (each rank
  reads its own file — no rank-0 memory bottleneck).
- Test: offline == online DistributedMesh (hash arrays) and identical solutions.

## 3. HYPRE extension
- [weakdeps] HYPRE → ext/XCALibreHYPREExt.jl; HYPRESolver <: AbstractDistributedSolver via
  IJ assembler; BoomerAMG default for pressure Poisson.
- Backend selection: linear_backend=:petsc|:hypre kwarg; SolverSetup untouched.
- Adjoint caveat: no transpose solve — SPD Poisson self-adjoint; error otherwise.

## 4. AD / adjoint (optional add-on)
- ChainRulesCore.rrule for halo_exchange! (pullback = halo_exchange_adjoint! via
  unpack_add!, kernels exist since Phase 2); psolve_transpose! (KSPSolveTranspose) rrule
  for the distributed solve; rrules for pnorm/pdot/pmean (communication-free pullback).
- ChainRulesCore becomes a Distribute dep (tiny).
- Tests (test/distributed/test_adjoint.jl, n=2,4): adjoint identity 1e-12; distributed
  gradient vs serial reverse-mode 1e-6; FD spot-check.
- Risk: Enzyme/KA version pinning — pin tested versions; hand-written rrule fallback.

## 5. Docs
- Workflow, cluster setup (MPIPreferences system binary, CUDA-aware env — reuse
  dev/gpu_native_setup.md), F32 (needs own env: dev/petscenv_f32 pattern — PETSc wrappers
  are precompile-time per-preference), CI notes, limitations (laminar-only until §1E,
  no cross-partition periodics yet, Windows serial-only, 1-layer halo ⇒ no least-squares
  gradients).

## Exit criteria
p-twins deleted with all distributed gates green + serial perf flat; turbulence case runs
distributed; offline == online; HYPRE matches PETSc; AD identities green (if built);
docs build.
