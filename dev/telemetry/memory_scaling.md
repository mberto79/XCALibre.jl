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

## P1-M28-S7 in-house element containers (option B)

- Strict class: 1t and 2D bitwise equal to the S1 baseline; 8t 10.1 and GPU 11.1 significant figures (their own rerun noise).
- Compile, same session, s (`compile_s`, `~/.cache/xcal_m28/{s7,base_s7,aos_s7}`): motorBike 1t base 18.06/17.94, S7 21.56/22.42/22.02 (+22%); 2D base 21.41, S7 27.40 (+28%, one sample each). Same as S4 (21.5, 27.5), so B recovered nothing.
- The base itself drifted 14.2 → 18.0 s (1t) since S1: the +51%/+67% quoted at S4 compared a loaded-machine sample against an older base; the same-session layout cost is +22-28%.
- AoS control (S7 source with the element arrays left as plain vectors): 1t 17.58/17.87, 2D 21.27, equal to base. The per-field layout, not its container type, is the whole rise.
- SnoopCompile (2D whole script, sum of exclusive inference by method): base 22.6 s, S7 29.7 s (+7.1 s, covers the run! compile rise); `KernelAbstractions.__run` +1.7 s, `SIMPLE` +0.8, `turbulence!` +0.5, rest spread over mesh-carrying methods. Julia-level inference/optimisation, not LLVM.
- MPI n=4 bitwise with fresh parts; `test_offline.jl` + `test_partition.jl` 4/4 at n=2,3 (48.9 s).

## P1-M28-S9 lazy element view (refused, reverted)

- Compile, same session, s: base 1t 16.64/16.57, 2D 20.15/20.14; S9 1t 20.49/21.73 (+27%), 2D 25.82/27.58 (+33%). No better than S7 (+22/+28); base drifted again 18.0 → 16.6 within the day.
- Strict class broken at 1t (Ux 9.9 figures), 2D bitwise: motorBike mesh columns `faces.e/delta/weight`, `cells.centre/volume`, `face_gDiff` hash differently from S7 on the same load, so a 3D reader read a lazy view after mutating its array in place. Diff kept at scratchpad only; not landed.

## P1-M31 screens on the S7 source (same-chain base each)

- One bounds check per element access, columns read `@inbounds`: 2D 24.65/25.61 vs base 19.80/19.88 (+27%), 1t 20.11/20.69 vs 16.40/16.53 (+24%). No change.
- AoS vectors behind a trivial `AbstractVector` wrapper with a user-defined `getindex`: 2D 19.59/20.17 vs 20.12/20.13, 1t 16.36/16.60 vs 16.56/16.64. Equals base.
- Mesh2 type tree (2D, type nodes/depth/string length): AoS 65/5/417, wrapper 103/6/668, S7 112/5/765; `ScalarField` carries the mesh. The wrapper is nearly as large as S7 and costs nothing.
- SnoopCompile S7 vs base: methods only in S7 total 0.15 s; the +7.1 s is the same caller methods inferring slower (`__run` +1.7, ModelPhysics +1.25, Solvers +1.0, e.g. the KOmegaSST model constructor 0.18 → 0.49 s).
- P1-M31-S1 (Discretise schemes, BC functors and bodies, boundary interpolation, Calculate kernels read columns; diff `dev/archive/patches/p1-m31-s1-column-reads.diff`): 2D 24.92/26.71 vs base 19.95/20.23 (+28%), 1t 20.49/21.04 vs 16.61/16.71 (+25%); 1t and 2D bitwise. Snoop delta vs base 6.76 s (S7 6.68): Discretise +0.60 (S7 +0.46), Calculate +0.35 (+0.37), `__run` +1.88, turbulence! +0.42, KOmegaSST constructor +0.31; the discretise kernel body itself +0.02.
- AoS wrapper also carrying the (unused) columns as a tuple: 2D 27.75/28.33 vs 20.11/20.17 (+39%), 1t 22.18/22.36 vs 16.62/16.61 (+34%). Carrying the column arrays in the mesh type is the cost, whether or not anything reads them.

## Flat mesh experiment (columns as top-level Mesh2/Mesh3 fields)

- Shape: per-field arrays as `Mesh3` fields (`face_centre`, `face_area`, `cell_volume`, `node_coords`, ...), same-typed arrays share one of 9 type parameters; `mesh.faces`/`cells`/`nodes` rebuilt on demand as views, no solver code changed. Diff: `dev/archive/patches/p1-m31-flat-mesh-columns.diff` (not landed).
- Compile s (base AoS / S7 containers / flat), same session: 1t 16.65,16.79 / 22.84,21.87 / 12.12,13.08; 2D 20.08,20.36 / 28.04,27.57 / 13.43,14.33; 8t 27.66,27.47 / 40.63,42.14 / 18.38,18.44. Mesh load 1t 2.7 / 4.0 / 3.7 s.
- 20-iteration run s: 1t 11.15,11.05 / 10.75,10.37 / 9.64,10.05; 8t 6.27,7.80 / 5.68,5.91 / 5.49,5.91. Flat 1t and 2D bitwise to the S1 baseline, 8t 10.0 figures. GPU, MPI and the suite not run.
- GPU (20 iterations, same session): flat before fix run 7.47/7.09 s vs base 4.10/4.53; S7 7.47 (S4 StructArrays 4.65). CUDA profile: GPU busy equal (256 vs 261 ms per 3 iterations), extra time on the host in `initialise_writer(VTK)` iterating a host copy of the mesh whose type was not inferred (container `adapt` closed over `to`; `typeof(Array)` is `UnionAll`). Also per call: discretise scalar kernel 5.67 vs 3.50 ms, vector 7.38 vs 4.94 ms; other kernels faster.
- After type-stable adapt (explicit column adapts, element type carried over; patch updated): GPU run 4.24/4.11 vs base 4.08/4.32 (parity), GPU compile 23.4 vs 26.6 s; 1t compile 13.6 vs 17.4, 2D 15.5 vs 20.8; 1t run 10.3 vs 11.4 s; GPU 10.7 figures, 1t and 2D bitwise.

## P1-M31-S5 flat mesh landed (same-chain base each)

- Compile s (flat / AoS base): 1t 12.61,13.41 / 17.03,17.50; 2D 14.69,14.99 / 20.58,20.86; GPU 22.37,23.17 / 26.23,26.44; 8t 18.29.
- 20-iteration run s: 1t 9.76,10.49 / 11.57,11.81; GPU 4.17,4.40 / 4.45,4.05; 8t 6.42. 8t pinned 100-iteration run 17.84,17.44 / 20.28,20.62.
- Accuracy vs `m28_baseline`: 1t, 2D, MPI n=4 (fresh parts) bitwise; 8t 10.5, GPU 10.3 figures. `test_offline.jl`, `test_partition.jl` at n=2,3 pass; `test_mesh_conversion.jl`, `unit_test_laplace.jl` pass.

## P1-M31-S7 type tags as one-element vectors

- Compile s (S7 / AoS base, same chain): 1t 13.38,13.52 / 17.35,17.01; 2D 15.02,15.22 / 20.43,20.32; rerun after tag fix 1t 12.94, 2D 14.76, GPU 23.26.
- Accuracy vs `m28_baseline`: 1t, 2D, MPI n=4 (S5 parts) bitwise; GPU 11.3 figures. `test_offline.jl`, `test_partition.jl` at n=2,3, `test_restart.jl` at n=1,2 (`dev/petscenv_stock`), `test_mesh_conversion.jl` pass.

## P1-M31-S6 column reads in the discretisation path

- GPU discretise per call, ms (scalar / vector, `gpu_profile.jl`, same session): S6 5.90 / 6.90; S7 HEAD 5.87 / 7.60; AoS base 3.50 / 4.95.
- PTX `__local_depot` per thread, discretise kernels (flat / AoS base): 3984 and 3480 B / 2304 and 1944 B; kernel parameter counts equal, so the rise is the by-value argument structs (mesh plus each term's field mesh) spilled to local memory.
- Compile s: 1t 12.94, 2D 14.85, GPU 23.89 (S5 12.6-13.4 / 14.7-15.0 / 22.4-23.2). Accuracy vs `m28_baseline`: 1t, 2D, MPI n=4 bitwise; 8t 10.8, GPU 10.3 figures; 8t run 5.87 s.
- Docs build green (after moving the `@define_boundary` helpers above its docstring). Suite files pass: `test_physical_boundary_conditions.jl` (updated to the new BC signature), `test_reconstruct.jl`, `unit_test_wall_function_averaging.jl`, `test_potential_flow.jl`, `unit_test_laplace.jl`, `test_mesh_conversion.jl`, 3D cascade periodic, rotating flat plate MRF, Taylor-Couette, oscillating cylinder, compressible fixedHeatFlux, fixedT, compression corner, 2D EFM.

## P1-M31-S8 kernel argument diet

- Discretise kernels take mesh-free terms, sources, `prev` and `rho_prev` plus `cells`, `faces`, `cell_faces`, `cell_neighbours`, `cell_nsign`; `model` and `mesh` no longer passed; phi keeps only `face_gDiff`.
- PTX `__local_depot` per thread (scalar / vector): 408 / 880 B (S6 3480 / 3984, AoS base 1944 / 2304).
- GPU discretise per call, ms (scalar / vector, `gpu_profile.jl`, same session): 0.66 / 1.73; AoS base 3.50 / 4.94. GPU 20-iteration run_s 3.79 (S6 4.20).
- Compile s: 1t 11.98, 2D 13.37, GPU 21.78 (S6 12.94 / 14.85 / 23.89). Accuracy vs `m28_baseline`: 1t, 2D, MPI n=4 (parts_s7_4) bitwise; GPU 10.6 figures.
- Suite files pass: `unit_test_laplace.jl`, `test_physical_boundary_conditions.jl`, oscillating cylinder, BFS Crank-Nicolson, rotating flat plate MRF, heated cylinder, KOmega fixedT, multiphase mixture, `unit_test_wall_distance.jl`; Smagorinsky errored in `bounding_box` (`get_backend` on a `FaceArrays` view, a flat-layout regression since S5), fixed to read `face_nodes`.

## P1-M28-S6 close (HEAD 6cf5ae70, flat column mesh, Int32)

Scratch drivers: copies of the benchmark `cpu_i32.jl`/`mpi_i32.jl`/`motorBike_gpu.jl`/`motorBike_profile.jl`/`kernels.jl`/`footprint.jl` in `~/.cache/xcal_m28/close/` (fresh `mesh_close.jld2`, offline parts, rows to `times.txt`); one command per point, unpinned clock, balanced profile, as the Int32 baseline.

| mode | cores | S6 close s | Int32 baseline s |
|---|---|---|---|
| threads | 1 | 135.62 | 162.90 |
| threads | 8 | 61.37 | 77.95 |
| MPI | 1 | 142.39 | 169.41 |
| MPI | 8 | 53.58 | 69.66 |
| GPU wg 32 | - | 17.17 | 22.4 |

- Refit from n = 1, 8 (`s(8)` 1.619): threads C 45.4 B 90.2 (baseline n = 1,6,8 fit B 119.3); MPI C 69.8 B 72.6 (baseline B 98.3). B −24%/−26%; C near the diagnosis fits (threads 34, MPI 64), the two-point fit is not directly comparable.
- GPU workgroup, 500 iterations, s: 32 17.17, 64 17.39, 128 17.63, 256 17.65; 32 stays the default.
- Isolated kernels on the real mesh object, 1t / 8t ms (speedup): face interpolation through `mesh.faces` 0.734 / 0.098 (7.5x, was 1.72x); cell gather 2.318 / 0.326 (7.1x, was 2.20x); CSR SpMV Int64 1.672 / 0.310 (5.4x), Int32 8.3x.
- Main-thread 8t profile, 100 iterations (10,627 samples): `wait` 50%; serial Krylov vector work (BLAS `axpby!` 11%, `bicgstab!` body 7%, `axpy!` 3%, `dot` 2%) ~22%; string building for progress output (`print_to_string`, `join`, `_string_n`) ~8%. Raw: `~/.cache/xcal_m28/close/profile_close_8t.{txt,jlprof}`.
- Footprint (Int32 serial mesh, column sizes): 223.6 MB (baseline AoS ~230 MB); `cell_nsign` Int8 2.1 MB (was 8.5 MB); face columns 136 MB, of which `face_centre`/`face_normal`/`face_e` 80 MB.
- Full serial suite by file via `suite_file.jl` (46 files, runtests.jl's list; 13 at S8, 33 here): all pass. Distributed gate: n=2,3 10/10 in 2m53s, n=6 `test_turbulence_sst_wallfn.jl` 1/1 in 56 s (3m59s together; 5m04s after S2, D167).
