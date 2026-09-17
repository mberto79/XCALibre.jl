# Strong scaling, distributed backward-facing-step, Cg + Jacobi, Float64, CPU

Probe: `dev/scripts/scaling_probe.jl`. Per-iteration cost is `(t100 - t3) / 97`, which cancels
compilation and setup. One rank per physical performance core, `--bind-to core --map-by core`,
BLAS pinned to one thread, offline-partitioned, no output written. Environment
`dev/petscenv_stock`: stock `PETSc_jll` + `OpenMPI_jll`, no preferences file, no shell exports.
`mhz` is the busiest core's clock sampled the instant the timed run ends.

## bfs_unv_tet_10mm, 68,243 cells, 2026-09-17

| ranks | s/iter | speedup | efficiency | MHz | clock-normalised efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0586 | 1.00 | 100% | 4643 | 100% |
| 2 | 0.0322 | 1.82 | 91% | 4100 | 103% |
| 4 | 0.0205 | 2.86 | 71% | 3500 | 95% |
| 6 | 0.0148 | 3.96 | 66% | 3300 | 93% |

Clock-normalised efficiency is `t1 f1 / (n t_n f_n)`: what the efficiency would have been had
every rank count run at the single-rank clock. It removes essentially all of the loss.

Final residuals agree across all four rank counts to thirteen significant figures
(p = 1.2781095690863e-3, Ux = 1.1747885585163e-3), so Q1 holds and the timing gap is cost,
not numerical drift.
