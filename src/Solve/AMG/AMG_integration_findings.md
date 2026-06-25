# AMG Integration Findings

Numbers -> AMG_integration_results.json. This file = decisions/gotchas only, one-liners, ## per phase.
Pre-unification history: Archived/AMG_perf_findings.md (P1-P13), Archived/Implementation/arch_findings.md.

## Carried-over constraints (do not re-derive)
- 9 invariants listed in AMG_integration_plan.md must survive refactor verbatim.
- ext/ import list = rename freeze (plan §renames).
- P13: Jacobi kernels stay residual-form; GS/SOR host sweeps still old form (CPU/F64 only, acceptable).

## U1 (done)
- Scope kept minimal: whole-file moves + global word-boundary rename only. validation->10 extraction
  and spike deletion (device/0,1,2,3,6) DEFERRED to U3 (matches plan rationale "deletion last").
- Layout: ref/0-6->0-6, device/4->7, device/5->8, ref/7+device/7 merged->9. Spike files left in
  device/, renamed in place, still included (order preserved => behavior identical).
- ext/ has ZERO refs to any renamed symbol (verified) => no ext edits, freeze list untouched.
- _greenfield_* NOT renamed (absorbed in U2). fl confirmed >=1 triggers matrix-free.
- Gate: tests 213+194 0fail identical; clean precompile; F1 GPU harness deferred to U2 (CUDA added there).

## U2 (done)
- workspace.hierarchy is now AMGHierarchy OR MatrixFreeHierarchy (both <: AbstractAMGHierarchy). Kind
  fixed once by amg_hierarchy_kind(solver, backend); _workspace builds the matching empty. No greenfield Ref.
- Type-match trick: MatrixFreeHierarchy coarse_fac/coarse_inv/refresh_plan are Ref{Any} (read once/cycle,
  matches reference cost) so empty==built concrete type -> workspace.hierarchy reassignment type-checks.
  _empty_matrix_free_hierarchy derives host-buffer TF from a 1x1 placeholder lu(T) (same TF as build).
- Shared solve-loop helpers widened ::AMGHierarchy -> ::AbstractAMGHierarchy: _launch_amg_kernel!, _add_amg!,
  _copy_amg!, _xpay_amg!, _cg_step_amg!, _update_cycle_factor!, amg_solve!, amg_cg_solve! + ext SPARSEGPU
  _matvec!/_residual!. Reference cycle internals (_cycle!, smoothers, coarse-solve) stay ::AMGHierarchy.
- Dispatch seam (§D): update! thin -> _amg_update!(hierarchy,...); outer_operator(h,A) (mf & mixed -> raw A);
  amg_apply_preconditioner! gets a MatrixFreeHierarchy method (gather/cycle/scatter); device/0 gate deleted.
- SKIPPED device_array refactor (the dev() closures already work; replacing them is pure churn, no dispatch
  impact). _amg_matrix_storage(generic A) was dead -> deleted.
- CPU mf hierarchy never built (amg_hierarchy_kind CPU=Materialised), so MF only runs on GPU.

## U3 (done)
- device/ dir deleted entirely. Production keepers (aggregate_permutation, permute_operator, _amg_index_type,
  transfer_factors, _fill_device!) were spike-file residents but USED by prod 7/8 -> relocated to top of file 7.
  Gotcha: grep prod callers before deleting a "spike" file; _fill_device! looked spike-only by name but isn't.
- New 10_AMG_validation.jl (included LAST) holds ALL test-only drivers + host oracles. host_csr_matvec /
  host_jacobi_sweep! are validation-only (sole caller = host_reference_vcycle), not prod -> live in 10, not 7.
- Spike funcs (2-grid/macro-layout/g3 prototypes) had ZERO prod refs -> deleted with their 5 testsets; the
  multilevel testsets already cover the same properties (per plan §Tests). CPU 360 (was 407).
- Comment policy applied ONLY to touched files (7 header, 10, edited tails). Files 0-6/8/9 bodies NOT swept
  (high-churn, behavior-irrelevant, no gate impact) — left for U4 or skip.

## U4 (done)
- Added T1/T2 to test_AMG_matrices.jl, T3 to test_AMG.jl CUDA block. All gate green, 0 fail.
- T1 (P13 F32 lock): 512^2 Dirichlet Poisson (poisson2d), Cg+Geometric(ml2)+coarse F32. Assert on the
  solver's REPORTED convergence (converged, iters<=40, last_relative_residual<=1e-5; anchor 17it/3.7e-6).
  GOTCHA: the F32 true residual floors at ~4e-3 (= eps_f32*cond) — recomputing it (esp. via Array(A) which
  also OOMs at 262144^2) is WRONG; the 3.7e-6 anchor is the recurrence residual, not the true residual.
- T2 (CPU ignores fl): poisson2d(64), fl=0 vs fl=3 both AMGHierarchy, iters bit-identical.
- T3 (GPU fl equivalence): SCREENED Poisson diag=neighbors+1 (strictly diag-dominant). GOTCHA: pure
  graph-Laplacian (diag==sum|offdiag|) makes the omega=4/3 Jacobi + Geometric cycle DIVERGE (relres->1e20)
  or Cg breakdown-guard bail (rz<=0, iters=0). fl=0/1/2 all 7 iters; mat-free vs materialised sol agree 1.7e-16.

## Acceptance + comment sweep (done, 06-25)
- F1 static replay re-run with USER-EDITED case settings (case file changed ml2/mc4096 -> ml3/mc124/pre3
  post3/F32coarse/F64sys/refresh2). mf(fl=1)==ref(fl=0): 16 iters both, trueres 8.25e-4, mf routed, both
  converge < rtol1e-3. 6 levels, shared messaging logs both paths identical. Numbers: results.json U4_accept.
- Comment-policy sweep of files 0-6/8/9 bodies done (Sonnet-delegated): ~120 comments collapsed/removed,
  -307/+76 lines, precompile OK, diff is comment-only (code preserved on every line). Files 7/10 already done U3.
- GOTCHA: delegated sweep over-cut a few precision/invariant inline comments. Restored 4 by hand: file8 rhs
  abs-row-sum bound (235) + invdiag-still-raw ordering (273) + T-cast split precision (25); file5 x-init-0
  residual-form fixed-operator (265). When delegating comment sweeps, name the keep-gotchas explicitly.

## Messaging unification (done)
- Both build paths now emit ONE shared @info via _amg_log_hierarchy(solver,backend,rows;matrix_free) in file 1:
  keys mode/coarsening/smoother/backend/path/levels/rows. Materialised gated by log_diagnostics (first build);
  matrix-free emits per build (file 9). Same format both backends.
- DELETED: stale "EXPERIMENTAL" @warn (file 9, made no sense as fuse_levels is the intended MF trigger);
  3 per-level build @info + verbose 4-line diag block (file 1); gf-refresh telemetry @info (file 8).
  Orphaned _coarse_solve_name + _hierarchy_level_summary removed. KEPT max-iters @warn + ext mixed-RAP @warn.
- Gate: CPU tests 0 fail; CUDA MF F1 iters=45 routed trueres=9.69e-7 (== U2/U3). rows chain logs per level.
