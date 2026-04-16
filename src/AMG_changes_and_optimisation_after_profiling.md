# AMG Solver — Optimisation Log

Branch: `HM/Algebraic-Multigrid-Solver` | Case: F1 1.67M cells, RANS KOmega, CUDA GPU | Updated: 2026-04-16

> **REPORT RULE**: Keep this file ≤ 150 lines. Compress or discard old detail when adding new results.

---

## STATUS: DONE — Loop closed 2026-04-16

**Best achieved:** SA 3L Jacobi@44504, coarse_sweeps=50, P_RTOL=1e-4 = **×0.666 measured** (iter 3b).
**Target:** 0.60 — not reached. Algorithmic floor confirmed.

### Path to 0.60 requires:
- GPU sparse direct at n≈44K (cuDSS or AMGX) — IC0 triangular solves sequential on GPU, slower than 50 Jacobi sweeps
- Or: additional level of hierarchy with better preconditioning structure

### AMG Loop — Final Iteration Summary (2026-04-16)
| Iter | Change | Ratio | Notes |
|---|---|---|---|
| 3b | Global cache + nzval sync | **0.666** | BEST — update 32ms, solve 224ms, 14.2 PCG iters |
| 4 | coarse_sweeps=100 | 0.716 | Worse: per-sweep cost > iter savings; solve 287ms |
| 5 | P_RTOL=5e-4 (looser) | 0.719 | Worse: CG saves more from looser tol than AMG |

### Key lessons from loop:
- **P_RTOL tuning**: helps CG more than AMG in absolute terms (CG has more iters to cut) — dead end for ratio
- **coarse_sweeps=50 is optimal**: 100 sweeps regresses due to per-sweep cost
- **IC0 at coarsest (n=44K)**: GPU sv2 triangular solves sequential, 6-20ms/call vs ~5ms for 50 Jacobi — SLOWER
- **5L hierarchy with cache**: adds ~15ms/iter Galerkin, +9 PCG iters → net worse than 3L
- **CPU sparse direct**: CHOLMOD 33ms/call vs 5ms GPU Jacobi@50 — impractical

---

## What to do next (if continuing)

**Step J.4 — GPU-native sparse direct (cuDSS)**
- Replace 50 Jacobi sweeps with cuDSS sparse direct at n≈44K — stays on GPU, no PCIe cost
- Currently `_MAX_DENSE_LU_N=5000`; need to raise to 50000 and use cuDSS instead of LAPACK
- Expected: PCG iters 14 → ~5-8; coarse solve 5ms → ~1-2ms
- Note: ic02 (IC0) was ruled out — sequential triangular solve is slower than Jacobi

---

## Profiling Results

### AMG Loop Runs (2026-04-16)
Global cache (`_GLOBAL_AMG_CACHE`) eliminates amg_setup! cost from timed window.
nzval sync: `copyto!(fine.A.nzVal, A_device.nzVal)` each iter (cheap ~0.2ms GPU-to-GPU).
Best: 3L SA, 50 Jacobi sweeps, update_freq=2, coarse_sweeps=50, P_RTOL=1e-4 → ratio=0.666.

### Rounds 7b–15 (compact)
SA+smooth_P best (pre-cache): update ~175ms/iter, solve ~211ms/iter, 23.8 iters, ×0.79.
Round 14: coarse_sweeps=50 optimal (14.0 iters, ×0.77 measured / ×0.66 ss); update_freq=2 optimal.
Round 15: CPU sparse Cholesky (CHOLMOD) at n=44504 → ×2.23 (33ms/call AMD fill ≈20-50×).
Round 13: SA 5L LU@28 ×0.62 ss (but ×0.74–0.79 measured, high variance); RS no better than SA.

---

## Optimisations Log
| # | Change | Result |
|---|---|---|
| O2 | cuSPARSE SpMV for smoother/residual | ~4× faster than KA kernel — active |
| O3 | Lazy Galerkin update_freq=2 | ~45 ms/iter saved vs freq=1 — active |
| O5 | cuSPARSE SpGEMM for smooth_P Galerkin | 204ms (RS) — active |
| O6 | SA coarsening + unsmoothed fallback | 56ms update — active |
| O7 | Pre-alloc AP KA kernels | **REVERTED** — 6–23× regression |
| O9 | P-truncation `trunc_P` param | **CLOSED** — collapses SA hierarchy; 3× slowdown |
| O10 | `unsafe_free!` + lazy A_f32 buffer | 7ms saved; GC no longer bottleneck — active |
| O11 | Global AMG cache + nzval sync | 175ms→32ms update; ratio 0.809→0.666 — active |

---

## Steps Closed
| Step | Result |
|---|---|
| J.5 | P_RTOL tuning: dead end — loosening helps CG proportionally more than AMG |
| J.4 | IC0 at coarsest: GPU triangular solves sequential, 6-20ms/call > 5ms Jacobi@50 |
| J.3 | CPU sparse direct (CHOLMOD) impractical at n=44504: 33ms/call vs 5ms Jacobi |
| J.1/J.2 | coarse_sweeps=50 best; update_freq=2 optimal (freq=3 +3.5 iters) |
| I | SA 5L LU@28 = ×0.62 ss; RS no better; diagonal guard eps() enables deep hierarchy |
| H | post_sweeps=2 → ×0.80/×0.66 ss |
| G | unsafe_free! -7ms GC |
| E/D/C | sub-timers; P-trunc fails; KA 6–23× slower; GPU dense LU too slow |

---

## Files Modified (current state)
| File | Change |
|---|---|
| `src/Solve/AMG/AMG_0_types.jl` | `coarse_sweeps::Int` in AMG struct (default 50); `A_f32_nzval::Any` in LevelExtras |
| `src/Solve/AMG/AMG_6_api.jl` | `_GLOBAL_AMG_CACHE` + nzval sync in `update!`; `_MAX_DENSE_LU_N=5000`; phase timers |
| `ext/XCALibre_CUDAExt.jl` | `amg_rap_update_smooth!`: lazy A_f32 + `unsafe_free!` on temps |
