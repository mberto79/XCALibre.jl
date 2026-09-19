# Gate results

| Timestamp | Session # | Gate Name | Target Metric | Actual Result | Status | Commit |
|---|---:|---|---|---|---|---|
| 2026-09-17 | 0 | serial suite | all pass | 1544/1544 pass, exit 0 | PASS | 9ca6f2c2 |
| 2026-09-17 | 0 | distributed suite n=1,2 | all pass | 26/26 pass | PASS | 9ca6f2c2 |
| 2026-09-17 | 1 | distributed gate n=2 | 5 files pass within 300 s | 5/5 pass, 89.3 s wall | PASS | 3887cc8b |
| 2026-09-17 | 2 | documentation build | builds with no errors | 0 errors, distributed page in output | PASS | 99d03e04 |
| 2026-09-17 | 3 | serial suite (full, Pkg.test) | all pass, no reduction on 1544 | 1548 pass / 1 fail: test_halo.jl `using Random` undeclared in test/Project.toml | FAIL -> fixed, rerun pending | 101c5b4b |
| 2026-09-18 | 3 | serial suite (full, Pkg.test) | all pass, no reduction on 1544 | 1549/1549 pass, exit 0, 21m15s | PASS | 2374a1e7 |
| 2026-09-18 | 3 | distributed gate n=2 (under Pkg.test) | 5 files pass within 300 s | 5/5 pass, 186.7 s at a pinned 2200 MHz | PASS | 2374a1e7 |
| 2026-09-18 | 3 | example on stock binaries, clean dir | runs at n=2 and n=4, writes decomposed output | parts/, processor0-3/, XCALibre.foam at both | PASS | 101c5b4b |

## P1-M7 close, 2026-09-18

- Serial suite `julia --project=. -e 'using Pkg; Pkg.test()'`: 1555/1555 (1549 + new freeze-constructor checks), distributed gate 5/5 in 88 s. Run before D41 removed the `reuse` alias; the post-D41 constructor change was re-checked by a direct serial smoke test: 3/4, the failure being a wrong expectation (`GAMG(reuse=3)` does not throw; unknown keywords forward to PETSc).
- `test/distributed/runtests_mpi.jl --ranks=1,2 test_hypre.jl` on `dev/petscenv_stock`: pass at n=1 and n=2 after the rename.
- `julia --project=docs docs/make.jl`: 0 errors; 2 warnings (reference page size, deploy detection), both pre-existing.

## P1-M8 close 2026-09-18
- distributed gate (2 ranks, petscenv_stock): 5/5 in 97.7 s.
- docs build (docs/makeLocal.jl) with new distributed-page doctests at 1 rank: exit 0, doctests green.
- probes: 1-process PETSc 2D BFS 20 iters Jacobi/GAMG OK; itmax=1 no throw; DILU->bjacobi, Cgs->cgs, NormDiagonal via `-pc_type sor` OK.
- 2026-09-18 P1-M9: distributed gate (2 ranks, petscenv_stock) 5/5 in 99.7 s; `test_gpu.jl` at n=1,2 on `dev/petscenv` (CUDA PETSc) pass, 88 s and 141 s wall including compilation.
- 2026-09-18 P1-M10: gate 5/5 in 100.9 s; `test_gpu.jl` n=1,2 pass on CUDA PETSc; hypre, turbulence, turbulence_sst, ppiso, periodic, io, offline 7/7 at n=2 (stock); `test_f32.jl` n=2 pass (petscenv_f32 re-resolved: its manifest predated the `Logging` dependency). Error path (CUDA fields + stock PETSc) pass at n=1 on the M9 tree.
- 2026-09-18 P1-M11-S1: distributed suite 12/12 at n=2 (stock), test_gpu n=1,2 (CUDA PETSc), test_f32 n=2 pass.
- 2026-09-18 P1-M14: distributed suite 12/12 at n=2 (stock, Int32 library selected).
- 2026-09-18 P1-M13-S1/S4: distributed suite 12/12 at n=2 (stock).
- 2026-09-18 P1-M13: distributed suite 12/12 at n=2 (stock); docs build with doctests green.
- 2026-09-18 pre-close: examples at n=2 complete - 2D_cylinder_U_mpi (stock, 26 s wall, final p 1.45e-6), 3D_BFS_mpi (stock, 57 s wall, solve 16.1 s, final p 7.19e-5), 3D_cascade_mpi_GPU (CUDA PETSc, 167 s wall, final p 3.37e-10).
| 2026-09-18 | P1-M17 | documentation build | builds with no errors | 0 errors, setup section and @ref links resolve | PASS | fad309b5 |
| 2026-09-18 | P1-M17 | distributed gate n=2 + test_gpu n=1,2 | all pass | gate 5/5 in 106 s (petscenv_stock); test_gpu pass on petscenv, petscenv_conda, petscenv_conda_ompi | PASS | fad309b5 |
- 2026-09-19 P1-M18 close: distributed suite 12/12 at n=1 and n=2 (`dev/petscenv_stock`); `test_gpu.jl` n=1,2 green on `dev/petscenv_conda_ompi` (host hypre: BoomerAMG guard errors) and `dev/petscenv` (CUDA hypre: BoomerAMG runs on device); S11 fault fixed (D85); `git diff main --stat` free of result files; docs build 0 errors at S1 (d91c67b8); M18-close rebuild 2026-09-19 exit 0, 0 errors (pre-existing size and deploy warnings only).
- 2026-09-19 P1-M19-S1: gate at n=2 and n=3 (`test/distributed/gate.jl`, five files) 10/10 in 196 s wall on `dev/petscenv_stock`, clock unpinned, against the Q2 bar of 300 s; `test_halo.jl` and `test_partition.jl` also green at n=5 and n=8 with the stock hydra launcher and no oversubscription flag.
- 2026-09-19 P1-M19 close: `test_ghosts.jl`, `test_perf.jl` (7 exchanges, 6 all-reduces per 2D SIMPLE iteration) and `test_invariance.jl` (spread 2.0e-6/2.7e-6/2.9e-6 vs n=1, bar 1e-4) green at n=1,2,3,5 on `dev/petscenv_stock`.
- 2026-09-19 P1-M20 close: `test_partition.jl`, `test_periodic.jl`, `test_offline.jl`, `test_assembly.jl`, `test_psimple.jl` 20/20 at n=1,2,3,5 on the rewrite; part digests identical to 3818c35b (`dev/telemetry/extraction_cost.md`).
