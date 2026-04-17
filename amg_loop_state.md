# AMG Optimization Loop — State File

Managed by `amg_loop.sh` + headless Claude. Do not manually edit during a run.


## Goal
AMG runtime ratio **< 0.60x** vs Cg+Jacobi baseline (F1 1.67M cells, RANS KOmega, CUDA GPU).

## Current Iteration: 3

## Change Made (Iter 3)

Replaced failed WCycle Config 4 with `coarse_sweeps=25` (half of current 50).
- AMG_6_api.jl: added `Dinv_Tc`/`b_Tc` allocation gated on `opts.fine_float == Float32` (harmless, needed for future)
- F32 fine smoother was attempted first but CATASTROPHICALLY FAILED: 1000 PCG iters (diverged).
  Root cause: F32 rounding breaks the SPD property of the AMG preconditioner — PCG requires SPD.
  F32 fine smoother path is CLOSED.
- coarse_sweeps=25: converges (16.5 PCG iters in smoke test vs 12.0 for cs=50).
  Hypothesis: F32 coarse with 25 sweeps may be sufficient (old log tuned at F64).

## Key Finding (Iter 2)
WCycle catastrophically failed: 161 PCG iters, ratio 6.456. Exponential recursion cost. CLOSED.

## Key Finding (Iter 1)
F32 vs F64 coarse V-cycle: 1.36x solve speedup — fine F64 SpMVs dominate.
Ratios: VCycle F32=0.655, F64-coarse=0.730.

## What to Try Next (after Iter 3 results)
- If cs=25 ratio ≤ 0.60: target met — set as new default
- If cs=25 ratio improves over cs=50 (0.662): keep and try cs=15
- If cs=25 ratio worse than cs=50: coarse_sweeps=50 is optimal → STATUS: EXHAUSTED
  (cuDSS at coarsest is the only remaining path but requires major new implementation)

## Closed Paths
- WCycle: exponential GPU recursion cost (2^n_levels calls)
- F32 fine smoother: breaks PCG SPD requirement → diverges (1000 iters)
- F64 coarse: 1.36x slower than F32 coarse, no benefit
- IC0 at coarsest: triangular solves sequential on GPU (from old loop)
- CPU sparse direct (CHOLMOD): 33ms/call vs 5ms Jacobi (from old loop)

## History

Iter: 0 (baseline)
Ratio: 0.653 | Solve: 218.85ms | PCG: 13.8 | Notes: SA 3L VCycle, smooth_P, coarse_sweeps=50

Iter: 1 (F64 coarse comparison added as Config 3)
Ratio: 0.655 | Solve F32: 235ms | Solve F64: 319ms | PCG F32: 14.9 | PCG F64: 15.4
F32 V-cycle solve speedup vs F64: 1.36x | Wall speedup: 1.11x

Iter: 2 (WCycle F32 added as Config 4 — FAILED)
Ratio VCycle: 0.662 | WCycle: 6.456 | WCycle PCG: 161 iters vs 15.8 VCycle
WCycle closed. Config 4 replaced with coarse_sweeps=25 test.
