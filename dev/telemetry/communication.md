# Communication rounds and reductions (P1-M22)

Machine: this laptop, `dev/petscenv_stock`, Julia 1.13, CPU, clock unpinned. Counters: `test/distributed/test_perf.jl` (2D BFS, laminar SIMPLE; 3D budgets are one higher each). Hashes: `dev/scripts/mem_probe.jl worker` on 10 mm offline parts, 2 iterations, Jacobi. Raw logs: `~/.cache/xcal_m22/` (not tracked).

## baseline (HEAD 8b769eb0)

- Per 2D SIMPLE iteration: 7 exchanges, 6 all-reduces (3D: 8, 8).
- `test_perf.jl` laplace halo allocation: 4016 B at n=2, 6544 B at n=3.
- Hashes: 10 mm n=2 `2a522e3ce43de833`, n=4 `dbc3c69ab48b3394` (the latter matches `fixed_rank_memory.md`).

## S1: one schedule per mesh, tags per width

- Counters unchanged (7, 6); halo allocation 2928 B at n=2 (−1088), 5456 B at n=3 (−1088); hashes identical at n=2 and n=4.
- Gate 10/10 in 204 s; `test_halo`, `test_ghosts`, `test_f32`, `test_perf` green at n=2,3.
- `test_gpu.jl` n=1,2 on `dev/petscenv_conda_ompi` green (2/2, 251 s).

## S2: one width-3 exchange for the momentum components

- Exchanges per 2D iteration 7 → 6 (3D 8 → 6), all-reduces unchanged; `check_ghosts` on U after the vector solve zero; hashes identical at n=2 and n=4; gate 10/10 in 200 s; halo, ghosts, f32, perf green at n=2,3.
- `test_perf.jl` vector-solve allocation 47264 → 49104 B at n=2 (two extra kernel launches); budget 98304.
- n=8 hash not run: language server resident (2.8 GB) and no 10 mm n=8 parts; n=4 stands in, as R8 makes them one bar.
- `test_gpu.jl` n=1,2 green (2/2, 254 s).

## S3: one all-reduce per equation for residuals

- All-reduces per iteration 6 → 2 (3D 8 → 2); exchanges 6; hashes identical at n=2 and n=4; laplace `residual` allocation 368 → 304 B; gate 10/10 in 200 s; halo, ghosts, f32, perf green at n=2,3; `test_gpu.jl` n=1,2 green (254 s).

## S4: one width-4 exchange for rD and Hv (SIMPLE body)

- Exchanges per iteration 6 → 5, all-reduces 2; hashes identical at n=2 and n=4 on 10 mm parts regenerated at part format 2 (format 1 parts are refused at the header); gate 10/10 in 200 s; offline, io, halo, f32, perf green at n=2,3.
- PISO keeps a separate exchange for rD and one for Hv per corrector: `H!` runs once per corrector, so rD cannot share the first one without moving `interpolate!(rDf, rD)` into the corrector loop.
- `check_ghosts` zero on rD and Hv after the paired exchange (`test_ghosts.jl` now mirrors the body) at n=2,3; `test_gpu.jl` n=1,2 green (255 s).

## S5: persistent requests

- Halo allocation 2928 → 2736 B at n=2 (5456 → 5200 at n=3); what remains is the pack/unpack launches, so the `test_perf.jl` budget drops to 512 + 3072 per neighbour.
- `dev/scripts/halo_bench.jl` at 10 mm n=8, 200 reps × 10 alternated rounds, medians: width 3 persistent 16.57 µs vs fresh 16.89 µs; width 1 11.89 vs 12.05 µs.
- Hashes identical at n=2 and n=4; gate 10/10 in 197 s; halo, ghosts, f32, perf green at n=2,3; `test_gpu.jl` n=1,2 green (256 s, covers the host-staged path).

## S6: overlap withdrawn on its upper bound

- 10 mm n=4 (`scaling_probe.jl worker`, 60 iterations, Jacobi, unpinned 3.5 GHz): 20.1 ms per iteration. Exchange at n=4 (`halo_bench.jl`): width 3 13.96 µs, width 1 12.26 µs; at n=8 width 3 16.57 µs.
- The two exchanges S6 would hide (∇p, rD+Hv) cost about 28 µs, 0.14 percent of an iteration at n=4 (about 0.3 percent at n=8), against a 3 percent bar; a full overlap cannot reach it here.
- A `_halo_begin!`/`_halo_end!` split of `halo_exchange!` alone raised its allocation 2736 → 2992 B and was reverted. 10 mm n=8 with PETSc does not fit beside the language server (memguard stop at 2446 MB free).

## S7: stream events withdrawn on its upper bound

- PETSc 3.25.5 (conda, CUDA) runs every device operation on one global current context with `PETSC_STREAM_DEFAULT` and a NULL handle, the CUDA legacy default stream; CUDA.jl's task stream is non-blocking, so the device drains are what order the two today.
- GPU residuals are not bitwise reproducible run to run (10 mm n=1: Ux `3.152013822883027e-5` vs `3.1520138228831386e-5`), so a race check on GPU must use a tolerance, not hashes.
- 5 mm n=1 baseline 51.6 ms per iteration (was 74.6 ms when the plan was written).
- Upper bound, A/B alternated in one session: Julia on the same NULL stream as PETSc and every per-solve drain removed, 57.3 / 57.2 ms vs 57.4 ms with drains (GPU warmer than at baseline); at most 0.3 percent against a 3 percent bar.
- Follow-up: MPICH printed `freeing inactive persistent request` at finalize for every schedule; each schedule now frees its requests from an MPI finalize hook, and the message is gone (`test_perf.jl` n=3, 0 lines).

## S8: owned-row kernel ranges withdrawn on its upper bound

- 10 mm n=4 CPU profile of 30 SIMPLE iterations (rank 0, 3438 samples): `discretise!` 9.2, `green_gauss!` 2.6, `div!` 0.3, `inverse_diagonal!` 0.2, `H!` 0.4 percent of samples, 12.7 together; `psolve!` 22.9; ghosts 3.5 percent of local rows.
- Skipping ghost rows saves at most 3.5 × 12.7 ≈ 0.44 percent of an iteration here (5 mm n=8: 2.9 percent ghosts, about 0.4); at n=64 (10.8 percent ghosts) about 1.4, which P2 can re-measure.

## M22 close

- Per 3D laminar SIMPLE iteration: exchanges 8 → 5, all-reduces 8 → 2 (2D: 7 → 5, 6 → 2); residual hashes bitwise identical at 10 mm n=2 and n=4 throughout; GPU n=1 51.6 ms per iteration at 5 mm (plan baseline 74.6 ms).
- n=8 pinned per-iteration time not recorded: 10 mm n=8 with PETSc exceeds memory beside the language server, and the overlap and ghost-row steps it was to judge were withdrawn on bounds below 0.5 percent.
