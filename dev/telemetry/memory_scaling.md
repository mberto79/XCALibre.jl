# Memory-traffic scaling, motorBike RANS (P1-M28, P1-M29, P1-M30)

Machine: i9-14900HX laptop, 8 P-cores, 14 GB, balanced power profile. Case: `~/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS` (353,830 cells, 1,108,598 faces, k-omega, CG+Jacobi pressure). Raw data and scripts: `data/scaling_diagnosis_2026-09-23/` there (`results.txt`, `kernels.jl`, `stream.jl`, `soa_test.jl`, `soa_gpu.jl`, `footprint.jl`). Source plan: `dev/archive/reviews/p1/memory-scaling-2026-09-23.md`.

## diagnosis (2026-09-23, before P1-M28)

- STREAM triad, pinned P-cores: 24.4 GB/s on 1 core, 39.5 GB/s on 8; ratio s = 1, 1.172, 1.586, 1.619 at n = 1, 2, 6, 8. Anything DRAM-streaming tops out near 1.6x.
- Model `T(n) = C/n + B/s(n)` fits every sweep within 5-7%. OpenFOAM PCG C 129 B 113; XCALibre MPI before face_gDiff C 184 B 113, after C 64 B 112; XCALibre threads after C 34 B 139 (seconds, 500 iterations). Lowering B is the only lever on the 8-core time.
- Kernel timings on the real mesh, 1t / 8t ms: face interpolation reading AoS `Face3D` 5.00 / 2.91 (1.72x); same from separate Int32/Float64 arrays 0.81 / 0.11 (7.48x); cell gather over `faces[cell_faces[fi]]` 8.99 / 4.09 (2.20x); CSR SpMV Int64 1.86 / 0.33 (5.62x); CSR SpMV Int32 1.66 / 0.20 (8.35x).
- `Face3D` with Int64 is 128 B (two cache lines; Int32 112 B); `faces` is 142 MB, about 4x L3; used fields sit on both lines, so each access pulls 128 B. OpenFOAM reads about 24 B per face from separate arrays.
- Threaded only: Krylov.jl 0.10 sends `Vector{Float64}` vector ops to OpenBLAS; `axpby!` is single-threaded there (2.4 s at 1t, 8.1 s at 8t in the profile), `kfill!` 1.3 to 2.0 s; about 13 s of solver vector work does not scale. Pinning OpenBLAS onto the Julia workers made a run 47% slower (21.6 to 31.7 s, 100 iterations). Interactive-thread layout is not a factor (`-t 8`, `-t 8,0`, `-t 7,1` all 21.5-21.7 s).

## StructArray probe, unchanged kernel code, Int32 mesh (`soa_test.jl`, `soa_gpu.jl`)

| kernel | AoS 1t / 8t ms | StructArray 1t / 8t ms | hand SoA 1t / 8t ms |
|---|---|---|---|
| face interpolation | 4.08 / 2.22 | 0.75 / 0.136 | 0.72 / 0.109 |
| cell gather | 6.93 / 4.27 | 3.23 / 0.906 | n/a |
| face interpolation GPU (RTX 4070, wg 32) | 0.270 | 0.060, identical results | n/a |

## Int32 mesh baseline (500 iterations, `integer_type=Int32`)

| mode | cores | Int32 s | Int64 s |
|---|---|---|---|
| threads | 1 | 162.90 | 176.61 |
| threads | 2 | 114.32 | 127.44 |
| threads | 6 | 83.77 | 96.93 |
| threads | 8 | 77.95 | 92.86 |
| MPI | 1 | 169.41 | 178.03 |
| MPI | 2 | 112.73 | 120.91 |
| MPI | 6 | 73.80 | 84.52 |
| MPI | 8 | 69.66 | 78.56 |
| OpenFOAM | 8 | 80.33 (rerun) | 81.66 |

Residuals match Int64 to 10+ significant figures. Refit n = 1, 6, 8: threads B 144.5 to 119.3 s, MPI B 116.1 to 98.3 s. 1-core gains sit inside the ±5% repeat noise; 6- and 8-core gains do not. GPU 500 iterations: 22.4 s.

## footprint, Int32 serial mesh (`footprint.jl`)

Mesh object about 230 MB, `faces` 124 MB, `cell_nsign` 8.5 MB (Int32, values ±1).

## P1-M28-S1 smoke baselines at 875c16bb (20 iterations, `dev/scripts/motorbike_smoke.jl`)

Files: `dev/telemetry/m28_baseline/{cpu1,cpu8,2d,gpu,mpi4}.{res,time}`. Env `~/.cache/xcal_m28/env` (copy of the benchmark `env_distributed`, XCALibre developed from the checkout).

- Reproducibility: 1t and MPI n=4 reruns bitwise identical; 8t and GPU reruns differ at about 10.5 significant figures (Ux worst; atomic boundary-face adds and GPU reductions), so those compare to ≥8 figures, never bitwise.
- First `run!` (1 iteration, compile-dominated), s: cpu 1t 12.3, 8t 12.8-13.5, 2d 17.5, gpu 21.9, mpi n=4 19.5. 20-iteration run, s: 1t 9.29, 8t 6.99-7.13, gpu 4.23-4.31, mpi n=4 3.85.
- MPI n=8 does not fit beside the language server (6.9 GB available); n=4 with offline parts is the MPI smoke point.
- An MPI run without `activate_multithread` took 241 s for 20 iterations instead of 3.9 s: every rank's BLAS spins on all cores (gotcha).

## P1-M28-S2 StructArray wrap (same session, base = 875c16bb worktree env)

- Strict class: 1t, 2D and MPI n=4 residuals and field hashes bitwise equal to the S1 baseline; 8t 10.7 and GPU 11.2 significant figures (inside rerun noise).
- 100 iterations, threads pinned to P-cores (`pinthreads(:cores)`), run s: 8t 19.18/18.85 base, 16.61/16.36 S2 (−13%); 1t 39.3 base, 35.0 S2 (−11%, one sample). Unpinned 8t runs land on E-cores and read 22.4 vs 19.6. 20-iteration MPI n=4 3.85 → 3.27 s.
- Compile (first `run!` after potential flow, `cumulative_compile_time_ns`), s: motorBike 1t 14.2 → 17.7 (+25%), 2D kwSST 16.5 → 23.3 (+41%), GPU first run 21.9 → 25.8 (+18%), MPI n=4 first run 19.5 → 21.6.
- Attribution (SnoopCompile, 2D case, whole script): exclusive inference 18.9 → 25.7 s, spread 1.3-1.5x over the same methods (`KernelAbstractions.__run` +1.6 s, `turbulence!` +0.5, `SIMPLE` +0.7); StructArrays internals 0.68 s. The rest of the compile rise is LLVM on per-kernel IR. Cause: every specialisation carrying the mesh type now carries 14 more column types.
- Footprint (Int32 motorBike serial mesh, `Base.summarysize`): 219 MB total, `faces` 118 MB, `cells` 16 MB, `cell_nsign` 8.1 MB.

## P1-M28-S3 part round trip and distributed gate

- `test_offline.jl` at n=2,3 (offline parts written and read equal online parts array by array, mesh type equal, cells/faces/nodes read back per field): 2/2 in 33 s.
- `gate.jl`: n=2,3 part 10/10 in 3m40.5s (3m14s before S2, D157); n=6 `test_turbulence_sst_wallfn.jl` 1m23.8s, killed by the 298 s outer timeout inside the gate, 7/7 per rank when run alone. Total 5m04s against Q2 (D167).

## P1-M28-S4 cell_nsign Int8

- Strict class: 1t, 2D and MPI n=4 bitwise equal to the S1 baseline (fresh format-4 parts); `test_mesh_conversion.jl`, `unit_test_laplace.jl`, `2d_godunov_supersonic_cylinder.jl` pass; `test_offline.jl` + `test_partition.jl` 4/4 at n=2,3.
- Compile (one sample, loaded machine), s: motorBike 1t 21.5 (base 14.2, +51%), 2D 27.5 (base 16.5, +67%), GPU first run 25.8. The extra mesh type parameter moved compile the wrong way; this triggered D168.
