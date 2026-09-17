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
every rank count run at the single-rank clock. The numbers are CONSISTENT WITH throttling
explaining most of the loss; they do not establish it. The correction assumes the work is
CPU-bound and scales linearly with clock, which sparse CFD only partly is, so it over-corrects
— the 103% at two ranks is that over-correction showing, not a superlinear gain. Rank count
and thermal load rise together in this experiment and cannot be separated from this data.
Halo exchange and the solver wrapper are unmeasured here, not ruled out (D8).

The clock figure is one sample of the busiest core taken after the closing barrier, during
cooldown; 4643 MHz on a 5.6 GHz core says the sample is late. It is an indicator, not a
measurement.

Final residuals agree across all four rank counts to thirteen significant figures
(p = 1.2781095690863e-3, Ux = 1.1747885585163e-3), so Q1 holds and the timing gap is cost,
not numerical drift.
