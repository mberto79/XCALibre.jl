# AMG Optimization Loop — State File

Managed by `amg_loop.sh` + headless Claude. Do not manually edit during a run.

## Goal
AMG runtime ratio **< 0.60x** vs Cg+Jacobi baseline (F1 1.67M cells, RANS KOmega, CUDA GPU).

## Current Iteration: 5  (2026-04-16)

## Current AMG Config
File: `F1-fetchCFD_Minimal/amg_loop_profile.jl` — `[AMG Config]` section.

```julia
AMG(coarsening=:SA, smooth_P=true, coarsest_size=50000,
    post_sweeps=2, coarse_sweeps=50, update_freq=2)
```
3L hierarchy: [1677158, 584946, 44500]. Global workspace cache + nzval sync active.
P_RTOL=1e-4 (reverted from 5e-4 — P_RTOL tuning is a dead end).

## Latest Results
Iter 3b (best): ratio=0.666, update=32ms/iter, solve=224ms/iter, 14.2 PCG iters.
Iter 4 (coarse_sweeps=100): ratio=0.716 — worse (per-sweep cost > iter savings).
Iter 5 (P_RTOL=5e-4): ratio=0.719 — worse (CG saves more from looser tol than AMG).

## History
| Iter | Change | Ratio | PCG iters | Notes |
|---|---|---|---|---|
| 0 | coarsest_size=100 → 5L LU@28 (unintended) | 0.880 | 19.5 | Step I diag guard eps() enabled deeper hierarchy |
| 1 | coarsest_size=100→50000 to restore 3L Jacobi@44500 | 0.868 | 13.3 | ~156ms/iter inflated by amg_setup! in timed window |
| 2 | AMG warmup: pre-build hierarchy before Config 2 timing | 0.809 | 15.7 | Warmup ran in separate workspace — amg_setup! still hit in Config 2 |
| 3a | Global workspace cache (`_GLOBAL_AMG_CACHE`) | 0.825 | 26.4 | Cache returned but fine_level.A pointed to stale warmup matrix — wrong SpMVs |
| 3b | + nzval sync in update! when fine.A !== A_device | **0.666** | 14.2 | update 175→32ms; solve 254→224ms; cache working correctly — **BEST** |

## Final Assessment

**Target 0.60 not reached.** Algorithmic floor is **~0.666** (3L SA, coarse_sweeps=50, P_RTOL=1e-4).

All parameter space exhausted:
- **Hierarchy depth**: 5L worse with cache overhead (+15ms Galerkin, +9 PCG iters)
- **coarse_sweeps**: 50 optimal; 100 regressed (cost > savings)
- **update_freq**: 2 optimal; 3 adds +3.5 PCG iters
- **IC0 at coarsest**: GPU triangular solves sequential (sv2), 6-20ms/call vs 2.5-5ms for 50 Jacobi sweeps — SLOWER
- **CPU sparse direct**: 33ms/call CHOLMOD vs 5ms GPU Jacobi at n=44504 — way worse
- **P_RTOL loosening**: helps CG more than AMG in absolute terms; ratio flat/worsens

**Path to 0.60**: requires better coarse solver — cuDSS or AMGX for on-device sparse direct at n≈44K.

## STATUS
DONE — best achieved ratio: **0.666** (iter 3b)
