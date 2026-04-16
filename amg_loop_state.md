# AMG Optimization Loop — State File

Managed by `amg_loop.sh` + headless Claude. Do not manually edit during a run.

## Goal
AMG runtime ratio **< 0.60x** vs Cg+Jacobi baseline (F1 1.67M cells, RANS KOmega, CUDA GPU).

## Current Iteration: 1

## Change Made (Iter 1)
Added Config 3 to benchmark: AMG with `coarse_float=Float64`, identical settings otherwise.
Purpose: quantify actual speedup from Float32 vs Float64 coarse levels (mission asks for this comparison).

## Next step (Iter 2)
Check Config 3 vs Config 2 solve times:
- If F64 coarse >> F32 coarse (>1.5x): Float32 coarse IS the bottleneck → optimize coarse path further
- If F64 coarse ≈ F32 coarse (<1.1x): fine-level Float64 SpMV dominates V-cycle → next step is Float32 fine smoother (use `A_f32_nzval` buffer in update! + smoother dispatch)

WARNING: This step is sensible, however, the run failed with error: "ERROR: LoadError: AssertionError: coarse_float=Float64 does not match _tc_vec_type result Float32; add GPU type-mapping methods for the desired coarse type
Stacktrace:
  [1] _workspace(amg::AMG{JacobiSmoother{Int64, Float64, Vector{Float64}}, VCycle}, A::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}, b::CuArray{Float64, 1, CUDA.DeviceMemory})
    @ XCALibre.Solve ~/Julia/XCALibre.jl/src/Solve/AMG/AMG_6_api.jl:95"

Action: Fix the error and modify the workflow/prompt to ensure you always run a smoke test following code changes to ensure the code runs. 

## STATUS
Iteration 1: Measurement run — added Float64 coarse comparison to amg_loop_profile.jl

## Latest Results (Iter 0 baseline)
AMG F32-coarse: 36.12 s (722.5 ms/iter) — ratio 0.653
Update: 32.24 ms (galerkin=30.87ms dominant), Solve: 218.85 ms, 13.8 PCG iters

## History

Iter: 0 (baseline)
Change: N/A — first run
Ratio: 0.653
PCG iters: 13.8
Notes: SA 3L smooth_P=true, coarse_sweeps=50, update_freq=2, coarse_float=Float32
