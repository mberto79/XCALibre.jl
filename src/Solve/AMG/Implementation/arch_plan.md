# AMG GPU Impl Archive — Plan

Scope: AMGSolver(), VCycle, Geometric(). GREENFIELD (D0). All phases 0-6 + G1-G5 DONE.

## Phases
- Ph0: baseline + harness (bench/amg_bench.jl, poisson3d)
- Ph1: reference/ = git mv 0..7; device/0 gate skeleton
- Ph2: knob reconciliation; fuse_levels::Int added (default 0, opt-in)
- Ph3: device/1_AMG_macro.jl: AMGMacroLayout, build_macro_layout, refresh kernel
- Ph4: SELL-P REJECTED (cuSPARSE at 90-93% BW; near-uniform FVM rows; Int32 packing already in ref)
- Ph5: 5a(2_fused)·5b(3_fused_cycle)·B(4_ml_cycle)·C(top-k matfree Galerkin)·
       G1(5_refresh)·G2(zero-alloc coarse)·G3(fused gate,REJECTED precond)·G5(guard)·
       scale_correction·lean_refresh(a/b/c)·gate_flip(opt-in)·seam(7_integration)
- Ph6: cylinder CFD (14% slower)·F1 3-way bench (gf 2x slower, -928MB VRAM)
- Ph7: G1-G5 done; G6 (FAS gate) not started

## State as shipped
- _greenfield_implemented()=true; fuse_levels default 0 (opt-in); 189/189 tests pass
- Cg mode validated: F1 gf 156 vs ref 161; AMGSolver F1 gf 334 vs ref 262 (direction gap, open)
- VRAM: gf 518MB vs ref 681MB F1 ml=1 (-163MB); CFD F1 ml=2/mc=100/F32: gf 3510 vs ref 4438MB

## Key structs (device/)
- MFGreenfield: st(MFMLState) + plan(MFRefreshPlan) + perm + r_perm
- MFMLState: levels(MFLevel[]), coarse_inv/host_bufs, coarse_max_rows, scale_correction
- MFLevel: tmp/r/sc, inv_diag, diag_index, omega, coarse_fac (all mutable for refresh)
- MFRefreshPlan: finest_host, coarsest_csr; lean (no rap_dst/rap_scale — recomputed in-kernel)

## Open
- Performance: gf ~2x slower than ref on F1 (profile + fix is perf phase)
- G6 FAS gate (user decision: speed-SOTA fork with standalone-solver mode)
- Option-C-with-refresh (further VRAM lever if OOM binds after perf phase)
- Default-ON (after perf gap + AMGSolver direction gap resolved or accepted)
