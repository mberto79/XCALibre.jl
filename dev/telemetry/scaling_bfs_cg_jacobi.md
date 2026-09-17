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

## bfs_unv_tet_4mm, 1,320,368 cells, 2026-09-17

Single-rank cost was re-measured alone, with nothing else resident, after the sweep showed 12 of
14 GB in use: 1.998 s/iter against 2.096 in the sweep, with `pswpout` unmoved across the run, so
the sweep was mildly inflated but never swapping. The clean figure is the baseline below.

| ranks | s/iter | speedup | efficiency | MHz |
|---:|---:|---:|---:|---:|
| 1 | 1.998 | 1.00 | 100% | 4451 |
| 2 | 1.231 | 1.62 | 81% | 3800 |
| 4 | 0.766 | 2.61 | 65% | 3300 |
| 6 | 0.671 | 2.98 | 50% | 3200 |

The 2026-07 sweep of the same case with Cg and GAMG gave 100 / 85 / 67 / 55%. Jacobi reproduces
that curve, so the preconditioner's hierarchy is not what the efficiency loss is made of.

### Heated single-rank control

One rank on core 0 with the other five performance cores held busy by scalar spin loops
(`dev/scripts/heated_control.sh`), so rank count and thermal state stop rising together:

| condition | s/iter | MHz at end |
|---|---:|---:|
| one rank, machine idle | 1.998 | 4451 |
| one rank, five cores loaded | 2.807 | 2900 |
| six ranks, per-rank-equivalent `6 x 0.671` | 4.027 | 3200 |

Thermal state alone costs 40% (1.998 to 2.807) with no parallelism involved. Efficiency at six
ranks measured against the heated single-rank cost is 2.807 / 4.027 = **70%**, against 50%
measured against the idle machine. So roughly two thirds of the apparent scaling loss on this
box is the package clock falling, not parallel overhead.

The heaters are scalar spin loops: they reproduce the thermal load of six busy cores but consume
no memory bandwidth, so the remaining 30% still mixes halo exchange with bandwidth contention.
The heated run also ended at a lower clock than the six-rank run (2900 against 3200 MHz), so 70%
is approximate. Splitting the residual needs PETSc `-log_view` at two and six ranks; PETSc.jl
registers an `atexit` finalize, so the report prints without extra code.
