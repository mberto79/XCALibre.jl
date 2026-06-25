# AMG Unification Plan — merge matrix-free (device/) + materialised (reference/) paths

GOAL: one AMG module, two hierarchy kinds selected by dispatch. Trigger = fuse_levels (fl):
- fl=0 -> materialised hierarchy (current reference path), CPU and GPU.
- fl>=1 on GPU (+ Geometric + AMGJacobi, else fallback to materialised with @warn maxlog=1)
  -> matrix-free hierarchy; fused_top = fl-1 (fl=1 = Option B, fl>=2 also fuses coarse operators).
- CPU: fl ignored entirely (matrix-free would be slower on CPU). Silent, per directive.

NON-GOALS: no algorithm changes, no tuning. This is a refactor: behavior-preserving, measured.

## Invariants that MUST survive (each was a hard-won fix; see Archived/AMG_perf_findings.md)
1. V-cycle rhs aliasing: post-smooth rhs = (l==1 ? level.rhs : level.r) — P9, stronger than textbook.
2. Jacobi kernel stays RESIDUAL FORM x + w*Dinv*(b-A*x), full-row sum — P13, F32 correctness.
3. Refresh takes the LIVE device matrix; never route refresh through _amg_setup_matrix/_amg_setup_backend
   (those go via host BY DESIGN; doing so re-adds a 310ms D->H per outer iter) — P8.
4. sc-in-Cg uses flexible PR+ beta; sc default-on both modes — P10.
5. Split precision: finest level ALWAYS at working T; coarse levels at coarse_storage TS — P3.
6. coarse_refresh_interval: off-iterations do light refresh (finest gather + invdiag only) — P12.
7. Fused refresh: composed maps + exact-diag Gershgorin (|diag| + cross-aggregate abs-sum); warm-start
   lambda (plan.eig, 2 iters warm vs 5 cold) — P11.
8. RAP scatter atomic casts product to eltype(dst_nz) (CUDA has no mixed-type atomic_add) — P3r.
9. Host coarsest direct solve always F64 (SuiteSparse rejects F32).

## Target layout (replaces reference/ + device/; update AMG.jl includes)
0_AMG_types.jl                  <- reference/0 (+ MaterialisedAMG/MatrixFreeAMG tokens, workspace change §W)
1_AMG_setup.jl                  <- reference/1
2_AMG_coarsening.jl             <- reference/2
3_AMG_transfer.jl               <- reference/3
4_AMG_smoothers.jl              <- reference/4
5_AMG_cycle.jl                  <- reference/5 (drop greenfield branch in amg_apply_preconditioner!)
6_AMG_cg.jl                     <- reference/6
7_AMG_matrix_free_hierarchy.jl  <- device/1 keepers (aggregate_permutation, permute_operator) + device/4
8_AMG_matrix_free_refresh.jl    <- device/5
9_AMG_update.jl                 <- reference/7 + device/7 seam (dispatch, §D)
10_AMG_validation.jl            <- host oracles + validation drivers (test-only callers)
DELETE: device/0 (gate absorbed), device/2, device/3 (after moving _host_jacobi_sweep!,
_mf_transfer_factors, _fill_device! out), device/6, device/1 leftovers (AMGMacroLayout,
build_macro_layout, refresh_macro_values!, macro_layout_spmv_error — spike-only; verify w/ grep).

## Rename table (no abbreviations; do mechanically in U1, incl. test files)
Structs:
  MFLevel -> MatrixFreeLevel ; MFMLState -> MatrixFreeHierarchy (absorbs MFGreenfield, §S)
  MFRefreshPlan -> MatrixFreeRefreshPlan
Production functions:
  _build_mf_ml -> build_matrix_free_hierarchy ; _mf_ml_hierarchy -> build_galerkin_operators
  mf_ml_cycle -> matrix_free_cycle! ; _mf_smooth! -> smooth_level!
  _mf_smooth_matfree! -> smooth_fused_level! ; _mf_apply_operator! -> apply_fused_operator!
  _mf_is_matfree -> is_fused_level ; _mf_coarse_solve! -> solve_coarsest_level!
  _mf_coarse_correction_scaled! -> apply_scaled_coarse_correction!
  _mf_transfer_factors -> transfer_factors ; _macro_permutation -> aggregate_permutation
  _permuted_operator -> permute_operator ; build_mf_refresh_plan -> build_refresh_plan
  refresh_mf_ml! -> refresh_matrix_free_hierarchy!
  _greenfield_build!/_refresh!/_update! -> absorbed into update! dispatch (§D)
  _greenfield_apply_preconditioner! -> amg_apply_preconditioner! method on MatrixFreeHierarchy
  _greenfield_guard -> inline @warn in build ; _use_greenfield_amg/_greenfield_implemented -> deleted
Validation (move to 10_AMG_validation.jl):
  _host_ml_vcycle -> host_reference_vcycle ; _host_jacobi_sweep! -> host_jacobi_sweep!
  _host_csr_matvec -> host_csr_matvec ; mf_ml_cycle_spike -> validate_matrix_free_cycle
  mf_ml_convergence -> validate_matrix_free_convergence ; mf_ml_cycle_allocs -> measure_cycle_allocations
  mf_ml_topk_error -> validate_fused_operator ; mf_ml_refresh_error -> validate_refresh
  mf_ml_refresh_convergence -> validate_refresh_convergence ; greenfield_solve_spike -> compare_amg_paths
  _frozen_rap_oracle -> host_frozen_rap_oracle
Kernels: keep _amg_* names except expand mf: _amg_mf_restrict_pos_kernel! -> _amg_matrix_free_restrict_kernel!,
  _amg_mf_prolong_{pos,set}_kernel! -> _amg_matrix_free_prolong_{add,set}_kernel!,
  _amg_bmar_kernel! -> _amg_residual_in_place_kernel!, _amg_jacobi_mf_kernel! -> _amg_jacobi_correction_kernel!.
DO NOT RENAME symbols imported by ext/*.jl (CUDA/AMD/oneAPI): _matvec!, _residual!, _prolongate_add!,
_amg_jacobi!, _amg_finalize_device_levels, _amg_finalize_transfer_plans, _refresh_coarse_level!,
_level_jacobi_omega, _launch_amg_kernel!, _amg_weighted_diagonal_correction_kernel!, AMGHierarchy,
AMGLevel, AMGMatrixCSR, AMGJacobi, AMGRAPPlanCPU, _refresh_coarse_operators!, _refresh_level_device!,
_refresh_coarse_cpu!, _amg_setup_backend, _amg_setup_matrix, update_preconditioner!, _m, _n
(grep each ext file's import block first; renaming any of these requires lockstep ext edits).

## §S MatrixFreeHierarchy struct (merge MFMLState + MFGreenfield + plan ownership)
Fields (existing MFMLState fields plus): refresh_plan (Ref or typed field; struct change => REPL RESTART),
cell_permutation_device (VI Int32), residual_permuted (VT, cycle storage type), nrows::Int, nnz::Int
(pattern guard), is_symmetric::Bool (compute on host build like reference _is_symmetric — Cg gate reads it),
last_cycle_factor::Float64, workgroup::Int. Keep levels::Vector{MatrixFreeLevel} ABSTRACT eltype
(mixed precision makes levels heterogeneous — do not over-parametrize).
Provide _empty_matrix_free_hierarchy(backend, T, TS) mirroring _empty_hierarchy so the workspace can
hold a typed-but-unbuilt hierarchy; update! populates on first call (same flow as reference).

## §W Workspace typing
_workspace(solver, b) picks the kind ONCE: amg_hierarchy_kind(solver, get_backend(b)) and constructs the
matching empty hierarchy; AMGWorkspace{H} stays concretely typed, no hot-loop dynamic dispatch.
amg_hierarchy_kind(solver, ::CPU) = MaterialisedAMG()
amg_hierarchy_kind(solver, backend::KernelAbstractions.GPU) = fl>=1 && Geometric && AMGJacobi ?
  MatrixFreeAMG() : MaterialisedAMG()   # config check stays one small predicate; backend split = dispatch
Invariant (assert in update!): hardware.backend == hierarchy backend kind chosen at workspace creation.

## §D Dispatch map (replace every backend/path if-statement)
- update!(ws, A, solver, config): thin entry -> _amg_update!(ws.hierarchy, ws, A, solver, hardware)
  with methods for AMGHierarchy (current reference body) and MatrixFreeHierarchy (current
  _greenfield_update! body: build on first call/pattern change via nrows+nnz, else refresh with
  coarse_refresh_interval light/full).
- amg_apply_preconditioner!(z, hierarchy, solver, r): two methods; DELETE the greenfield[] Ref check.
- solve_system! outer operator: outer_operator(h::MatrixFreeHierarchy, A) = A;
  outer_operator(h::AMGHierarchy, A) = _amg_mixed_precision(h) ? A : h.levels[1].A.
- device_array(::CPU, v) = copy(v); device_array(backend, v) = Adapt.adapt(backend, v) — replaces all
  `backend isa CPU ? copy(v) : Adapt.adapt(...)` closures (build, refresh, integration).
- AMGHierarchy.greenfield Ref{Any} field: DELETE (and its _empty_hierarchy slot).
- Data-dependent branches stay as-is (is_fused_level inside cycle, coarse_inv === nothing): these are
  per-level runtime facts, not backend selection. Optional later: coarsest-solver dispatch types.
- _update_cycle_factor!/_reset_residual_history! etc.: make hierarchy-generic (both kinds carry the
  fields they touch).

## Comment policy (apply in U3)
Strip ALL multi-line narrative headers (file-top essays, phase references, plan citations). Keep ONLY
one-liners stating non-obvious constraints: the 9 invariants above at their code sites, UMFPACK F32->F64,
OnDevice>max_rows fallback, KA launch fixed-alloc note in measure_cycle_allocations. Nothing else.

## Tests (U4; runtests.jl already includes both files)
Update renamed symbols. Replace deleted-spike tests (fused_2grid, build_macro_layout,
macro_layout_spmv_error, g3_block_smoother_gate, 2-grid validate calls) with the multilevel
equivalents already covering the same properties. ADD:
- T1 F32 regression (locks P13): 512x512 Poisson (5-point, Dirichlet), all-F32 CPU, Cg ml2 mc1024
  coarse_storage=F32: assert converged && iters<=40 && rel<=1e-5 (anchor: 17 it, 3.7e-6).
- T2 CPU ignores fl: same problem F64, fl=3 on CPU -> workspace hierarchy isa AMGHierarchy AND iters
  == fl=0 run exactly.
- T3 GPU equivalence (inside CUDA.functional() block): F64 device system, fl=0 vs fl=1: iters within
  +-1, solutions rel-agree < 1e-8; fl=2 converges (exercises fused refresh+cycle).
Keep: oracle parity, refresh error/convergence, sc fused-vs-materialised equivalence, F32/F64 Cg parity.

## Acceptance gates (run after EVERY phase; baselines in AMG_integration_results.json "baseline")
- Both test files pass (record new counts when spike tests are replaced in U3).
- Standalone F1 harness (results file has the snippet): F64 ml2 mc4096 rtol1e-6 gf 45 == ref 45;
  all-F32 rtol1e-4: 31 == 31; per-cycle F64 <= 8 ms; full refresh <= 30 ms.
- Final (after U4): F1 static solve wall-time within noise of baseline; optional 50-iter SIMPLE smoke
  (casesXCALibre/F1-fetchCFD_Minimal/f1_amg_vs_jacobi.jl, background+redirect) within ~3% of 0.6349 s/it.

## Phases
U1 Mechanical: move files to new layout, apply rename table, update AMG.jl includes + test symbol names.
   NO logic edits. [Sonnet-delegable: give it ONLY this plan §layout+§renames+ext-list + acceptance]
U2 Dispatch + struct merge (§S/§W/§D): absorb MFGreenfield, workspace-kind selection, delete gate file
   + greenfield Ref. REPL RESTART after struct change. [main model]
U3 Delete spikes, consolidate validation file, comment cleanup per policy. [Sonnet-delegable + review]
U4 New tests T1-T3, full acceptance run, record results, update MEMORY.
Rationale: rename while old tests still pin behavior; structural change second; deletion last.

## Gotchas (enforced; history in Archived/)
- AMG struct/plan struct change -> REPL restart (Revise cannot redefine structs).
- CUDA = LOCAL-ONLY dep: never commit Project.toml/Manifest.toml (git restore before staging).
- Refresh validation MUST run on CUDA: Atomix on CPU silently promotes mixed-precision atomics (false pass).
- ONE Julia process at a time (14GB host RAM); GPU = RTX 4070 8GB, watch VRAM.
- Device test matrix via ModelFramework._build_A(backend, findnz(parent(A))..., n) — bare CuSparseMatrixCSR
  (the live CFD type); adapt() on SparseXCSR densifies (20TiB trap).
- Don't hand kernel internals to Haiku; Sonnet minimum for kernel-adjacent edits.
