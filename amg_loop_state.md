# AMG Optimization Loop — State File

Managed by `amg_loop.sh` + headless Claude. Do not manually edit during a run.

ACTION: The results from the current iteration (iteration 2) are already available. You can start this iteration by analysing the results and moving on to work on "what to try next" (below)

## Goal
AMG runtime ratio **< 0.60x** vs Cg+Jacobi baseline (F1 1.67M cells, RANS KOmega, CUDA GPU).

## Current Iteration: 2

## Change Made (Iter 2)

Added Config 4 to benchmark: AMG WCycle with F32 coarse (same settings as Config 2 but `cycle=WCycle()`).
Purpose: test whether W-cycle reduces PCG iters enough (15→8-10) to offset higher per-cycle cost (~1.33×)
and reach the 0.60 target. Config 2 (VCycle) preserved for continuity of BENCHMARK_RATIO.

## Key Finding (Iter 1)
F32 vs F64 coarse V-cycle solve speedup: **1.36x** (235ms vs 319ms) — fine-level F64 SpMVs dominate.
Overall wall time speedup: 1.11x. 2x expectation not met because 1.67M-cell fine level ≫ 44K coarse level.
Ratios: VCycle F32=0.655, F64-coarse=0.730. Fine-level work dominates the V-cycle.

## What to Try Next (after Iter 2 results)
- If WCycle F32 ratio ≤ 0.60: target met — swap Config 2 to WCycle()
- If WCycle ratio 0.61-0.63: combine WCycle with F32 fine smoother
- If WCycle ≥ VCycle (diminishing returns with inexact coarse): skip WCycle, try F32 fine smoother instead
  - F32 fine smoother path: add `Dinv_Tc::TcVec` + `b_Tc::TcVec` to LevelExtras fine level;
    cast b→F32 once before sweep loop; use r_Tc/tmp_Tc as ping-pong; reconstruct A_f32 SPARSEGPU in-place

## Previous Recovery (Pre-launch Iter 2)
Fixed `_tc_sparse_type`/`_tc_vec_type` to accept Tc type arg (was hardcoded Float32).

## Recovery (Pre-launch Iter 3)
Fixed `amg_rap_update_smooth!`: derive `Tc = eltype(P.nzVal)` for SpGEMM (was hardcoded Float32).
All 3 configs smoke-tested OK.

## History

Iter: 0 (baseline)
Ratio: 0.653 | Solve: 218.85ms | PCG: 13.8 | Notes: SA 3L VCycle, smooth_P, coarse_sweeps=50

Iter: 1 (F64 coarse comparison added as Config 3)
Ratio: 0.655 | Solve F32: 235ms | Solve F64: 319ms | PCG F32: 14.9 | PCG F64: 15.4
F32 V-cycle solve speedup vs F64: 1.36x | Wall speedup: 1.11x
Key finding: fine F64 SpMVs dominate; 2x expectation not met
