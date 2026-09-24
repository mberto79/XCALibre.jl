# P1-M36 - cheap per-iteration wins (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R7, R13. Governing decisions: D200, D202, D207, D208. Source: `dev/archive/reviews/p1/pre-merge-review-2026-09-24.md` § Performance (C1-C9). Not a merge blocker (user, 2026-09-24).

## Problem, quantified

Shares of the 8t motorBike loop from the M30 close profile (`~/.cache/xcal_m28/close/profile_m30_8t.txt`, thread 1, ~1.9 ms/sample); workers idle ~1/3 of the time.

- C1: `residual()` (`src/Solve/Solve_1_api.jl:454-469`) does an extra SpMV and two serial `sum`s, 6× per iteration: 476 samples ≈ 7.7%; isolated 0.58 ms/call, threaded partials 0.41 ms.
- C2: `discretise!` zeroes `nzval` before the kernel (`src/Discretise/Discretise_2_generated_distretisation.jl:41,118`) though the kernel assigns every entry when no BC extends the pattern (motorBike nnz = n + 2·n_ifaces): ~0.45 ms/call, ~2%.
- C5: serial main-thread passes: double `p` copy (`src/Solvers/Solvers_1_SIMPLE.jl:214-215`), `nut = k/ω` broadcast (`src/ModelPhysics/Turbulence/RANS_kOmega.jl:247`), wall scratch `fill!` (`RANS_functions.jl:89`): ~1-1.5%.
- C6 (GPU, guess): ~31 `KernelAbstractions.synchronize` per iteration (`src/Discretise/Discretise_5_apply_bcs.jl:48`, `Solvers_1_SIMPLE.jl:369,377,465,532`) plus residual `sum` syncs: 2-5%.
- D202: VTK `initialise_writer` builds output strings (~3 s per run) even with `write_interval=-1`.
- Deferred, not in this milestone: C3 relaxation Σ|a| from discretise, C4 fused Jacobi refresh (changes Uy/Uz bitwise), C7 BC kernel argument diet (GPU), C8 fused Krylov / gradient, C9 (serial analogue of D207, user decision).

## Approach

Remove work that produces nothing: no new data structures. C1 cheap form only (per-chunk partials through `_reduce_chunks` on CPU, one fused device reduction on GPU); the "reuse Krylov's residual" form is refused here because it changes the residual definition. C2 zeroes only when the equation's pattern was extended (flag set by `extend_matrix`). D202: skip `initialise_writer` when `write_interval ≤ 0`, or build it lazily on the first write.

## Configuration space

motorBike 1t, 8t, GPU, MPI n=4 (20 iterations, `~/.cache/xcal_m28/chain.sh` + `mpi.sh`) and 2D BFS (`2d`), plus a periodic case (cascade suite file) for C2's gate. Strict: residual histories bitwise at 1t, 2D and MPI n=4 (C1 threaded partials change summation order: then the R13 band, D197); GPU ≥ its own rerun spread.

## Steps

Expected 4 steps.

- [x] **P1-M36-S1** LANDED (D219). C2 + C5: zero `nzval` only for extended patterns; drop the duplicate `p` copy, `nut` and wall scratch via `xcal_foreach` - mechanism: the discretise kernel writes every entry of an unextended pattern - cost: none - verdict: bitwise 1t/2D/MPI; cascade periodic suite file passes; 8t run_s not slower.
- [-] **P1-M36-S2** WITHDRAWN (D220), refused on its bar. C1: residual numerator/denominator from one fused pass with chunk partials (CPU) and one device reduction (GPU) - mechanism: the residual is a reduction and reductions are chunk-ordered (D196) - cost: none - verdict: 1t bitwise when serial path is taken; 8t within the R13 band; 8t run_s improves beyond noise, else refuse.
- [x] **P1-M36-S3** LANDED (D221). C6 + D202: remove redundant `synchronize` calls (keep the one before host reads and inside the periodic BC method), skip VTK writer initialisation when nothing will be written - mechanism: KA launches on one stream are ordered; a writer that never writes needs no state - cost: none - verdict: GPU residuals within rerun spread, GPU run_s from `gpu_profile.jl` not slower; VTK output still written when `write_interval > 0` (writer test files).
- [x] **P1-M36-S4** LANDED (D222). close: 500-iteration 1t/8t/MPI 8/GPU timings with `~/.cache/xcal_m28/close/` drivers, one point per command; full serial suite.

## Exit criterion

S1-S3 landed on their bars; 8t 500-iteration time recorded against 48.1 s; no strict-class break.

## Open questions

- SETTLED (D221): skip; SIMPLE and PISO write nothing when `write_interval` is negative.
