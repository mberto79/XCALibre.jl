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
