# AMG Perf Findings

Numbers live in AMG_perf_results.json. This file = decisions/diagnosis only.

## P1 Root cause of the ~2x CFD gap (F1 ml=2/mc=100/F32) — MEASURED
- Gap is ITERATION COUNT, not kernel speed. gf is a weaker preconditioner.
- PCG iters to 1e-6: gf 273 vs ref 167 (1.63x). x per-cycle 1.15x = 1.88x ~ observed 2x.
- CORRECTED 2025-06-08 (clean 2x2 FR-PCG, results.scale_correction_2x2_frpcg): the gap is the DIRECTION
  GAP, sc-independent. ref production Cg = 167 is NO-SC (prior "ref uses sc in Cg" was FALSE; sc gated
  off in Cg for BOTH by design). 2x2: ref 167/126, gf 273/219 (nosc/sc). gf/ref = 1.63x nosc, 1.74x sc.
  -> sc does NOT close the gap. P3 (close direction gap) takes gf-nosc 273->~167 = ref default, no CG risk.
  -> P2 (sc) = SEPARATE additive ~25% lever for BOTH, but changes shared CG gate + nonlinear-M (FR) risk;
     deferred (see below). Direction gap = same OPEN issue as arch AMGSolver 334 vs 262; LIKELY a scaled-
     correction accumulation bug / missing normalization in mf_ml_cycle (18x norm, cos 0.996 vs ref).
- P2 status: sc DID converge under FR-PCG on F1 (no breakdown, faster), but that is ONE matrix; the
  documented FR fixed-SPD-M assumption (cg:363) still makes unguarded sc-in-Cg unsafe in general. Enabling
  it safely needs flexible CG (Polak-Ribiere+) AND touches the reference's gate too -> separate decision.

## P6 REGRESSION LOCALIZED (06-08, results.P6_regression_localized) — supersedes "refresh 7%" below
- The 1.5x CFD regression is the per-timestep REFRESH, NOT the cycle. Standalone build->refresh->solve on
  real F1 1.68M F64: gf iters == ref (88/79 vs 89/80) and refresh does NOT change iters (gf 88==88 build/
  refresh). gf per-CYCLE is even slightly FASTER than ref (8.45 vs 8.68 ms). So the inner loop IS frozen,
  zero assembly — directive #1/#2 premise REFUTED by measurement.
- The whole gap: gf refresh 205ms vs ref 30ms. _amg_matrix(A2) was 147ms (72%) doing a redundant full
  rowptr/colval D->H+Int extraction; the frozen refresh only needs values. FIX (device/5:176): nz2 =
  _cpu_vector(_nzval(A2)). 205->143ms (-30%), iters unchanged, oracle relerr ~1e-16.
- Remaining 143 vs 30: distributed (RAP scatter cascade ~80, values D->H 22, host gather 17.5, lambda
  recompute 12.4, coarsest LU 11.4). Each modest; deeper work (cache lambda? faster scatter?) for follow-up.
- SCOPE/CAVEATS (advisor): (1) REGRESSION REDUCED NOT ELIMINATED — fixed 62ms of a ~175ms refresh gap
  (143 vs ref 30). (2) Refresh time is convergence-INDEPENDENT so -60ms/outer-iter (ml2) TRANSFERS to
  CFD without re-running it; vs measured 0.42 s/it P5 gap it dents only part. (3) iter-parity gf 88==ref
  89 is rtol=1e-6 (ASYMPTOTIC) — does NOT transfer to the CFD's LOOSE per-outer pressure tol; cannot
  claim CFD iter-parity. (4) ml3 45% Ux divergence STILL OPEN — tight-tol parity does not rule out
  loose-tol early-iter behavior or an ml3 refresh bug; moots any ml3 wall-time compare (correct-vs-wrong).
  (5) numbers at mc4096; deploy is mc1024 (fix helps regardless). (6) refresh-vs-cycle split done via
  CUDA.@elapsed, NOT literal NVTX markers (same quantitative split the directive's NVTX was for).
  (7) clean before/after Δ is ml2 only (ml3 509ms was compile-polluted).

## P7 REFRESH GAP CLOSED — device finest gather (06-08, results.P7_finest_device_gather)
- The remaining P6 refresh gap (gf 143 vs ref 30, "RAP cascade ~80" + "values D->H 22" + "host gather
  17.5") was MOSTLY the finest host round-trip, NOT the RAP cascade. The CFD system matrix A2 is
  device-resident (_amg_setup_matrix = identity), yet refresh did _cpu_vector(_nzval(A2)) (D->H) +
  host gather loop + H->D copyto = ~47ms of pure data shuffling for values already on the GPU.
- FIX: move finest_value_map to device (CuArray Int32), gather on device with the EXISTING
  _amg_gather_kernel! (no new kernel). adapt(bk,_nzval(A2)) is zero-copy on a device A2 (verified
  ===), one H->D only for a host A2 (validation driver). Dropped finest_host + struct field.
- RESULT (same-harness apples-to-apples, F1 ml2 mc4096 F64): gf refresh 26.4/30.6 (min/med) vs ref
  30.1/35.7 -> gf refresh <= ref. The refresh REGRESSION (the whole gf-vs-ref gap per P6) is CLOSED.
- DIRECTIVE #1 REFUTED BY MEASUREMENT: RAP scatter cascade = 3.18ms, NOT ~80ms (the 80 lumped the
  finest H->D). Caching per-nonzero RAP maps (~52-104MB VRAM, the reverted phase5d approach) to
  optimize a 3ms cascade is pointless -> NOT implemented. Advisor §2/§3 gate (measure before storing
  maps) decided it: no headroom case, no time to gain.
- New refresh breakdown (gf 30ms): finest 0.94, cascade 3.18, smoother(invdiag+lambda) 13.2, coarse
  (D->H+LU+denseinv) 12.4. smoother is now the largest single component; gf already <= ref total so
  further work is past-parity tuning (deferred). lambda warm-start = the correctness-neutral lever.
- SCOPE: refresh component only (convergence-independent, transfers to CFD); does NOT prove end-to-end
  CFD wall-time parity. ml3 45% Ux divergence still OPEN+separate. mc4096 numbers; gather is O(nnz) so
  mc1024 unaffected. Correctness: F1 device oracle 4.75e-16, both AMG test files pass.
- WHY 30ms TRANSFERS (not just a clean-loop artifact): the old path allocated a fresh 68MB host array
  (_cpu_vector) EVERY refresh -> GC/host churn the spike paid (~the 143-vs-76 isolated delta) but the
  loop didn't. New path = ZERO host alloc on the hot refresh -> the isolated 30ms should hold in the spike.
- CFD CLAIM GUARD: refresh is no longer a contributor, but CFD wall-time will still regress ~1.3-1.4x
  from the untouched per-cycle 1.15x + loose-tol convergence (P5: ref ml2 0.842 vs gf 1.256 s/it; the
  ~113ms/it refresh was only ~part). "gf value = VRAM (3-3.8x)" stands. End-to-end needs the bg SIMPLE run.
- TRANSFER-RISK (fold into bg run, not separately checkable): zero-copy adapt verified on a
  CuSparseMatrixCSR (_build_A). Production passes a device SparseXCSR; _nzval(SparseXCSR) is a plain
  CuArray so adapt stays no-op, but assert _nzval(A) isa CuArray at the refresh call site in the bg run.

## Ruled OUT (measured, don't rechase)
- [SUPERSEDED by P6] Refresh "56ms = ~7%" was mc100/F32 and/or excluded host extraction. In real F1 F64
  mc4096 the refresh is ~200ms and IS the whole regression bucket. Directive#1's freeze/precompute-scatter
  still wrong (reverses VRAM win); the real refresh fix is values-only D->H (P6), already applied.
- Per-cycle speed: gf 4.62ms vs ref 4.03ms = 1.15x only. Our finest SpMV = 1.15x cuSPARSE (competitive).
- Launch-bound tail: L5-8 are launch-floor (~7us) but only ~3% of cycle (L1+L2 = 73% real compute).
  -> CUDA graphs / kernel fusion of small grids won't move the needle.
- Coarse solve precision: F32 GEMV inverse == F64 host LU (both 273). Not the cause.
- Operators/P/lambda_max: IDENTICAL gf vs ref (P is 1nz/row both; lambda matches to 3dp). Not it.

## Directive assessment (user's NEXT-SESSION levers vs measurement)
- #1 freeze projection indices / stream coeffs: optimizes refresh (7%), reverses VRAM win. Premise
  ("106ms recompute per iter killing us") FALSE — refresh is 56ms and 7%.
- #2 workgroup mult-32 / pack aggregates: kernels are @index(Global), wg=256 already mult32, 1 thread/row.
  No per-wg-per-aggregate structure. No-op here.
- #3 pad stencils / kill divergence: forces max-stencil work on irregular CFD mesh -> SpMV anti-pattern, hurts.
- #4 register/F32: config already F32. Marginal.
- None address the dominant lever (iteration count / preconditioner quality).

## Real levers (in priority)
1. scale_correction in Cg/preconditioner mode (273->219, ~20%). NOTE: standard Fletcher-Reeves pcg
   already tolerated sc here (both gf 219 + ref 167 converged). Open question = verify the PRODUCTION CG
   path is stable with sc enabled in Cg mode before flipping the gate (user-facing behaviour change).
2. Close the direction gap (219->167, ~24%). 18x norm + 0.4% dir error per apply -> hunt the scaled-
   correction accumulation bug. If fixable, closes most of the remaining 1.31x.

## P3 ROOT CAUSE (2025-06-08): F32 cycle ASYMMETRY breaks CG. NOT algorithmic. (results.p3_root_cause)
- Clean CG-preconditioner-mode probe (my frpcg = exact Fletcher-Reeves, validated: reproduces 273/167):
  F32 gf 273/ref 167; F64 gf 90/ref 99 (gf even BETTER). So the gap is F32-ONLY, mode-specific.
- The OLD "algorithmic + precision-independent direction gap" (gf 98 vs 61) was from AMGSolver STATIONARY
  mode, NOT the CG path the CFD uses. In CG mode F64 there is NO gap. Misdiagnosis corrected.
- IT IS A QUALITY DEFICIT IN gf's F32 CYCLE, NOT a CG-symmetry artifact (asymmetry is a SYMPTOM):
  - Mixed-prec (F64 outer + F32 cycle = CFD path): gf_FR 273, gf_FLEX(Polak-Ribiere) 284 (NO recovery),
    ref_FR 99, ref_FLEX 99 (control correct). FLEXIBLE CG FALSIFIES 'asymmetry-breaks-CG': a flexible/
    variable-M solver does NOT recover gf -> the outer Krylov cannot fix it.
  - Stationary(Richardson) F32 floors gf 4.3e-4 vs ref 2.2e-5 (20x): a quality/floor gap with NO CG.
  - gf F32 cycle is EQUALLY LINEAR/consistent as ref (additivity err 8.5e-8 both) and per-apply random-
    vector error is SIMILAR magnitude (gf 0.036/ref 0.033, 1.08x). So gf's M is a valid linear operator,
    just a WORSE F32 approximation of A^-1 - its error evidently hits the low-freq modes AMG relies on
    (random-vector per-apply err can't see that). The 1.6e-3 asymmetry is one face of this worse error.
  - F32-ONLY: in F64 gf 90 ~ ref 99 (gf even better). Note F64-outer helps REF a lot (167->99) but gf is
    STUCK at 273 either outer precision -> gf's F32 cycle quality is the hard ceiling.
- SOURCE NOT LOCATED. RULED OUT (all measured): coarse solve (gf host-LU-F64 -> SAME 273), transfer
  adjoint (~1e-7 ALL levels), operators (gf==ref Frobenius 0.0), omega (4e-8), linearity (8.5e-8 both).
  Earlier 'transfer accum amplified by smoothing' and 'asymmetry-breaks-CG' guesses are BOTH REFUTED.
- IMPLICATION/FIX: outer-loop fixes (flexible CG) are DEAD. Closing it needs higher-fidelity F32 cycle
  arithmetic (e.g. F64-accumulate SpMV/transfer reductions, storage stays F32) - UNTESTED, source-
  unlocated, uncertain, and costs the speed/VRAM parity. This is RESEARCH, not a quick fix. Meanwhile gf
  in F64 is production-equivalent (gf 90 ~ ref 99) WITH the 3-3.8x VRAM win - the deployable path today.

## Old reconstruction (now reinterpreted)
1.63x iters x 1.15x cycle = 1.88x ~ CFD 2x still holds as the F32 arithmetic, but the 1.63x is the
ASYMMETRY penalty, not an algorithmic quality gap. gf also pays gather+scatter ~+6%/CG-iter.

## P3 RESOLVED — split precision (finest F64 / coarse F32). MEASURED 06-08
- FIX: gf cycle now finest level A0 = F64 ALWAYS; coarse levels(>=2)+coarsest = coarse_storage.
  Code: _build_mf_ml Tl=l==1 ? T : TC; _greenfield_build! passes F64 matrix + coarse_storage=TS, r_perm=F64.
- DECISIVE (results.mixedprec_*): gf_f32m iters == gf_f64 iters in ALL 32 gf cells (poisson+F1, both modes).
  Mixed precision = full F64 cycle quality. F1 Cg: gf 124/90/89 vs ref 128/99/93 -> gf BEATS ref.
- The P3 F32 deficit is ENTIRELY in finest-level F32 arithmetic. Coarse-level F32 costs ZERO iters.
  CAUSAL CONTROL (results.mixedprec_causal_control, same harness, only precision differs): all-F32 finest
  = 400 iters NO convergence; finest-F64+coarse-F32 = 96 == all-F64 96. Airtight.
- VRAM (measured F1 ml1 mc1024): gf_mixed 348MB vs gf_f64 449MB (1.29x less); all-F32 ~281MB. Mixed costs
  +67MB for F64 finest A0 but keeps the bulk coarse-level win = full F64 quality at near-F32 VRAM.
- gf_f32m also FASTER than gf_f64 in wall time (F32 coarse = less data); beats ref_f64 time at mc1024.
  mc124+ml1 gf is slow -> deep-hierarchy launch overhead (mc1024 has LARGER coarsest yet is faster, so
  NOT the coarse solve). mc1024 strongly preferred for gf. Per-level omega gf == ref EXACTLY (dir #3 OK).
- ref_f32 == ref_f64 on F1 Cg (ref cycle precision-insensitive) — the deficit was always gf-specific.
- AMGSolver standalone: gf > ref iters (gf 147 vs ref 115 etc) — SEPARATE deficit, NOT precision
  (gf_f32m~gf_f64). Matches old P1 F64 98vs61. Cg is the CFD path; this is its own open item.
- Directive #6 reality: scale_correction is AMGSolver-only; in Cg it is OFF for both by design.
- Open follow-ups: (1) AMGSolver-standalone gf>ref iters (own item). (2) restrict-boundary
  F64->F32 accumulation left as-is (didn't need tightening; mixed already == all-F64).

## P3 FOLLOW-UP RESOLVED — mixed-prec device/5 REFRESH (transient path). MEASURED 06-08 (results.mixedprec_refresh)
- THE ONLY real bug was GPU-only: coarse RAP scatter (_amg_rap_scatter_recompute_kernel!) did an atomic
  add across the precision boundary (F64 finest src -> F32 coarse dst); CUDA atomic_add has no mixed-type
  method -> would throw. FIX: cast product to eltype(dst_nz) in the atomic (no-op for uniform precision).
- Two other split-precision touches were benign auto-converts (NOT errors), fixed for hygiene: coarse_inv
  rebuilt at TC=eltype(st.coarse_rhs) not finest T; _fill_device! zero at dst eltype.
- VALIDATED ON CUDA (the bug is invisible on CPU — Atomix promotes silently; CPU test alone = false pass):
  mixed relerr 3.4e-7 (=F32 coarse precision), F64 regression 4.4e-16. Refresh-then-converge: mixed iters
  == F64 iters (1600: 6==6; deep 14400/9-level: 5==5). Coarse F32 costs ZERO iters through the refresh too.
- New drivers device/5: mf_ml_refresh_error gains coarse_storage kwarg; mf_ml_refresh_convergence (build
  A1 -> refresh A2 -> stationary MF V-cycle on A2, reports iters). Tests: CPU in test_AMG_matrices G1,
  GPU guard in test_AMG.jl CUDA block. Mixed-prec gf is now CFD-deployable on the transient path.
- WHEN IT HELPS: transient CFD with coarse_storage=Float32 (per-timestep refresh). WHEN NOT: uniform
  precision unaffected (cast is no-op); coarsest staging stays F64 in the plan (matches build, harmless).

## P5 HIERARCHY DEPTH = the unmeasured gf-vs-production gap (CPU probe 06-08, results.P5_*)
- gf is LOCKED to Geometric coarsening (device/0:19). Production CFD ref uses SmoothAggregation.
- F1 1.68M coarsening rate: SA ~7.5x/lvl -> 4 levels. Geometric ml1 ~1.7x -> 15 levels; ml2 ~3.4x -> 7;
  ml3 ~6.6x -> 5 (~= SA depth). ml1 = ~4x more per-cycle kernel launches than SA.
- ALL prior gf-vs-ref validation (incl mixedprec 124-vs-128) used Geometric ml1 for BOTH = both 15-level.
  The 'ref' was NEVER production SA. gf-Geometric vs production-SA-Cg in real CFD = UNMEASURED.
- Existing mixedprec data already shows depth is a big lever: gf ml2 89it/659ms vs gf ml1 124it/1287ms (~2x).
  -> For deploy, use Geometric ml2 or ml3 (not ml1). NEXT: real-CFD wall-clock gf-Geo-ml{2,3} vs SA-Cg.
- WHEN IT HELPS: any gf config -> raise merge_levels to ~match SA depth. WHEN NOT: tiny meshes (few levels anyway).

## P5 DENSE COARSE BRANCH — user OOM hypothesis RESOLVED (NOT the cause)
- Dense coarse inv (device/4:100, Matrix+inv) builds ONLY when coarsest_n <= OnDevice.max_rows (default 512),
  and is REBUILT every refresh (device/5:208). Deploy cfg (Geo ml1 mc1024) coarsest=990>512 -> host-LU, NO dense.
- Crash was HOST RAM OOM: 14GB total, 2x 963MB-mesh Julia procs (REPL + bg sim) + VSCode. Run ONE proc at a time.
- REAL footgun (keep in mind, not hit by defaults): high OnDevice(max_rows) capturing a LARGE coarsest =>
  per-timestep O(n^3) re-inversion + dense n^2 alloc. Default 512 caps it ~2MB/cheap.
- Host-LU coarse branch (coarsest 990-4815 > 512) is the PRODUCTION per-cycle path and is the slow one
  (sync+host copy+ldiv!+copy, device/4:131-136) — still untested in the real solve loop.

## P5 MATRIX-FREE REGRESSES ~1.5x vs MATERIALIZED in REAL CFD (06-08, results.P5_matfree_vs_materialized_CFD)
- F1 real SIMPLE CFD 50 iters, F64 both, same Geometric coarsening. s/it (FULL outer-iter, incl U/k/omega):
  ref ml2 0.84 / gf ml2 1.26 (1.49x); ref ml3 0.74 / gf ml3 1.17 (1.58x). Pressure-only regression LARGER.
- MECHANISM NOT ISOLATED: no inner-iter counts (solve_history gone) -> can't tell 'slower kernels/same iters'
  from 'more iters/weaker matrix-free cycle'. ml2 Ux matches (1%) but ml3 DIVERGES (gf 3.97e-3 vs ref 2.74e-3,
  ~45%) -> matrix-free cycle NOT numerically identical at ml3; 'pure overhead' UNPROVEN.
- THE BIG LESSON: prior REPL 'gf beats ref' (1287<1585) measured the STATIC pressure solve in ISOLATION,
  excluding the per-timestep hierarchy REFRESH the CFD pays EVERY outer iter. In the real update!->solve
  loop, matrix-free LOSES. This is exactly the host-LU + full-loop path the caveat said was untested.
- ml3 < ml2 for both (shallower hierarchy faster). Plain Cg+Jacobi cheapest/outer-iter (0.64).
- WHEN gf HELPS: VRAM-bound (3-3.8x less VRAM -> bigger meshes). WHEN gf HURTS: wall-time when VRAM not limit.
- NEXT: inner-iter counts via standalone update!->solve loop (ws.iterations); device matrix via
  CuSparseMatrixCSR ctor NOT adapt (adapt densifies SparseXCSR -> 20TiB). THEN NVTX cycle/refresh split.

## P8 — the ~1.5x CFD DEFICIT ELIMINATED: hidden D->H in the refresh seam (06-08, results.deficit_2_rootcause)
- SUPERSEDES the P5 conclusion above. The deficit was NOT weaker iters and NOT the cycle. Decomposition
  (F1 bin system, CuSparseMatrixCSR on device): iters gf==ref at loose(1e-2) AND tight(1e-6) tol, ml2+ml3
  (directive#2 'iters drift' REFUTED). gf SOLVE is FASTER than ref. The WHOLE gap = refresh gf 342ms vs ref 27ms.
- ROOT CAUSE: _greenfield_refresh! called _amg_setup_matrix(A, _amg_setup_backend(backend)). GOTCHA:
  _amg_setup_backend(CUDABackend)=CPU (NOT identity!). _amg_setup_matrix(CuSparseMatrixCSR, CPU) = full D->H
  copy of all 8.5M nnz = 310ms/outer-iter. refresh_mf_ml! itself is 27ms (P7 right); the SEAM added the D->H.
- LIVE TYPE: GPU matrix is a BARE CuSparseMatrixCSR (.nzVal capital). SparseXCSR is HOST-ONLY (_nzval reads
  lowercase .nzval off a host SparseMatrixCSR). Bin e.A=SparseXCSR is the serialized host copy, NOT the live type.
  Proof: adapt is identity on a device CuArray, so real-CFD telemetry zero_copy=FALSE => A2 was host => D->H ran.
- FIX (device/7): pass live A straight to refresh_mf_ml! (drop _amg_setup_matrix in the REFRESH; BUILD keeps it
  for host aggregation). refresh gathers _nzval(A) device-direct in frozen fvm order; device/5 guard adapts only
  a host A (validation driver). VALIDATED standalone (19.9ms, converges) + CONFIRMED in CFD bench below.
- POST-FIX CFD (ITERS=50, zero_copy=true): gf ml2 1.205->0.8545 (ref 0.8291, 1.03x); gf ml3 1.113->0.7463
  (ref 0.7506, 0.99x = FASTER). gf residuals consistent (gf ml2 p 6.5e-4 < ref 8.7e-4), no ml3 divergence.
- NET: matrix-free gf is now WALL-TIME PARITY/slightly faster vs materialized AND keeps 3-3.8x VRAM win.
- LESSON: when reusing the reference setup helpers (_amg_setup_matrix/_amg_setup_backend) on the device path,
  they ROUTE THROUGH HOST by design (reference builds host-side). The matrix-free refresh must bypass them.

## P9 — Lever A SOLVED: the standalone iter gap was a CYCLE ALGORITHM mismatch (06-10, results.P9)
- Reference _cycle! ALIASES level.rhs at levels>=2: the recursion passes coarse_level.rhs as the rhs
  arg, then _residual!(level.rhs, ...) overwrites that SAME array -> ref post-smooths coarse levels
  against the pre-smoothing RESIDUAL, not the textbook restricted rhs. Level 1 is unaliased (textbook).
- gf implemented the textbook cycle (host-oracle-validated) — and that is the WEAKER variant here:
  gf standalone 151 vs ref 121. One-cycle outputs differed 13% maxrel even without sc (cos 0.9992).
- FIX: replicate ref in mf_ml_cycle + _host_ml_vcycle (post_rhs = l==1 ? lv.rhs : lv.r). gf standalone
  151->121 == ref; Cg 89==89 (unchanged); oracle parity 6e-16. Operators/omega/coarse were NEVER the
  issue (verified identical level-by-level). LESSON: validate the ALGORITHM against the reference
  implementation, not only against your own oracle of your own algorithm.

## P10 — Lever C: sc-in-Cg via flexible PR+ β (06-10, results.P10)
- sc was AMGSolver-only by the FR fixed-SPD-M argument. Enabled in Cg for BOTH paths with flexible
  Polak-Ribiere+ β (max(0,(rz₊-z₊·r_prev)/rz)); r_prev reuses workspace.correction (idle in Cg). Cost:
  1 copy + 1 dot per CG iter + 1 finest SpMV per cycle (Ac; the old residual recompute is dropped —
  down-sweep lv.r is reused, valid because lv.x is untouched in between).
- F1 static F64 ml2: 89->45 iters both paths (gf 0.44s, beats ref 0.54s); loose 1e-2: 30->15.
- DEFAULT CHANGE: scale_correction=true (ctor default) now ACTIVE in Cg mode — Cg iteration counts
  change for every AMG user (improve on F1; sc can be turned off per-solver). FR β also tolerated sc
  on F1 (same 45) but PR+ is kept: nonlinear M breaks the FR assumption in general, PR+ costs ~nothing.

## P11 — Levers B+D: fused-level device refresh + warm-start λ (06-10, results.P11)
- fuse_levels>1 transient path UNBLOCKED: composed per-fine-row maps (comp_dst Int32 + comp_s T, two
  reused device buffers, ~20MB F1) hop through the un-materialized block; exact fused invdiag by
  diagonal scatter; λ by power iteration through _mf_apply_operator!; ONE composed RAP scatter fills
  the first materialized level below the block. Guard removed: fused_top = fuse_levels-1.
- Fused Gershgorin GOTCHA: naive Σ|s_i s_j a_ij| inflates the floor (within-aggregate sign
  cancellation on the diagonal slot) -> omega 20-40% small. Fix: exact assembled |diag| + cross-
  aggregate abs-sum = exact for M-matrices, safe bound otherwise. Oracle 4e-16 values+omega.
- λ WARM-START (Lever D): persistent per-level eigenvector in MFRefreshPlan; warm refresh = 2 power
  iters (cold 5). Refresh 27 -> 20-24ms on F1 ml2. Drift 0 on repeat refresh.
- fuse>1 SPEED VERDICT (directive premise REFUTED by measurement): F1 static solve 45 it all configs;
  fuse1 0.443s / fuse2 0.657s / fuse3 0.860s — every fused-level apply routes through ONE FINE SpMV
  (Galerkin chain), so deeper fusing is strictly slower. VRAM beyond fuse1: only -33MB (F1). WHEN IT
  HELPS: extreme VRAM pressure only. WHEN NOT: any wall-time goal -> fuse_levels=1 stays the config.

## P12 — vs PCG+Jacobi (static; CFD bench in flight)
- F1 pressure system, cold start: tight 1e-6 AMG 45it/0.443s vs Jacobi-PCG 1078it/1.436s (3.2x);
  loose 1e-2 AMG 15it/0.192s (+~0.02 refresh) vs 378it/0.490s (~2.3x). CFD 50-iter run: f1_amg_vs_jacobi.jl.

## P12 FINAL — AMG beats PCG+Jacobi (06-11, results.P12_CFD_interval)
- Enabler stack: sc-in-Cg FCG (halved iters) + Lever A cycle fix + warm λ + NEW gf
  coarse_refresh_interval support: _greenfield_update! now mirrors reference 7_AMG_update semantics —
  off-iterations do a LIGHT refresh (finest gather + finest invdiag only, refresh_mf_ml! coarse=false);
  coarse cascade/λ/LU every k-th. Staleness is preconditioner-only (outer CG reads live A) — safe, and
  FCG tolerates the variation anyway.
- F1 SIMPLE 50-iter same-run: gf ml3 i4 rtolp1e-1 0.6349 s/it vs jacobi 0.6522 (-2.7%), p/Ux quality
  matched. gf i1 also faster (0.6476). rtolp1e-2 +4.4% buys 3.6x deeper p.
- HONEST CLAIM: loose-tol steady margins are within ~2-4% run-to-run variance -> "parity to slightly
  faster" at matched quality. DECISIVE wins: tight tolerance (static 3.2x), transient/PISO regimes,
  larger meshes (jacobi iters grow with conditioning; AMG's don't), VRAM (matrix-free 2.1x+).
- WHEN JACOBI STILL WINS: nothing on F1 at matched quality; generally smallest meshes where AMG build
  cost never amortizes, or extremely loose single-iter pressure solves on well-conditioned systems.
- Tests: test_AMG.jl 213/213 (GPU incl; F32/F64 Cg parity held despite nonlinear-M), matrices 194/194
  (old sc+fused @test_throws replaced by fused-vs-materialized sc equivalence test).

## P13 — F32 finest cycle quality SOLVED: diag-excluded Jacobi kernel form (06-12, results.P13)
- p3_root_cause CLOSED. Cause: _amg_jacobi_step_kernel! used (1-w)x + w*Dinv*(b-sigma_offdiag) —
  recombines two large independently-rounded terms, injecting ~ulp(|x|) noise per sweep. At F32 the
  preconditioner turns noisy/nonlinear at small residuals -> CG stall (F1 all-F32: stall 2.4e-4) and
  breaks flexible CG too (explains old 'FLEX no recovery'). FIX: residual form x + w*Dinv*(b-A*x),
  full-row sum (one kernel, one less array read, diag_index no longer needed by the sweep).
- Why ref-GPU was immune: CUDAExt _amg_jacobi! override (cuSPARSE residual + correction kernel) IS
  the residual form at every level. gf-GPU and ref-CPU shared the fused kernel -> IDENTICAL stall
  trajectories (98it/7.46e-4 nosc). The bug therefore also afflicted the REFERENCE CPU path at F32.
- Validation (F1 1.68M ml2 mc4096): all-F32 gf 31==ref 31 (true rel 1.07e-4, was stall 2.7e-4);
  F64 gf 45==ref 45 unchanged; ref-CPU-F32 106-stall -> 31; mixed gf 48/ref 51 ok; per-cycle
  7.32ms (<= 8.45 recorded); tests 213/213 + 194/194. Host oracle updated to same form.
- Small-problem repro IMPOSSIBLE in practice: Poisson 256², 1D n=8192 (kappa 7e6), jump-coeff all
  show NO old-vs-new difference (stationary Richardson is rate-limited, not noise-limited; small-n CG
  converges before noise matters). Regression lock = all-F32 full-solver CPU test, 512² Poisson
  rtol 1e-5: NEW kernel 17 it / 3.7e-6 (assert conv && iters<=40). F1-scale was the only true repro.
- WHEN IT HELPS: any F32/mixed hierarchy (gf all-F32, ref CPU F32). WHEN NOT: F64 (identical algebra,
  rounding-level diffs only); GS/SOR host sweeps still use the old form (CPU-only smoothers, F64 use).
