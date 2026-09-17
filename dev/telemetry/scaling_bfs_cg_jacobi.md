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

## XCALibre against OpenFOAM, bfs_tet_5mm, 499,503 cells, 2026-09-17

Same mesh, same session, same binding (`--bind-to core --map-by core`, the serial OpenFOAM run
pinned with `taskset` to match), same `(t100 - t3) / 97` window, and the same solver settings:
pressure relative tolerance 0.01 under-relaxed 0.2, velocity relative tolerance 0.1 under-relaxed
0.8. OpenFOAM 12 `foamRun`, PCG with diagonal preconditioning and PBiCGStab with diagonal;
XCALibre `Cg` with `Jacobi` and `Bicgstab` with `Jacobi` through PETSc.

| ranks | OpenFOAM s/iter | efficiency | XCALibre s/iter | efficiency | XCALibre speed advantage |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.186 | 100% | 0.650 | 100% | 4.9x |
| 2 | 1.936 | 82% | 0.372 | 87% | 5.2x |
| 4 | 1.001 | 80% | 0.223 | 73% | 4.5x |
| 8 | 0.526 | 76% | 0.174 | 47% | 3.0x |

XCALibre is faster in absolute terms at every rank count, and at eight ranks it is still three
times faster than OpenFOAM. Its parallel EFFICIENCY is better at two ranks and worse at four and
eight, and the gap at eight ranks is 29 points.

Part of that is arithmetic rather than a defect: efficiency is measured against a single-rank
baseline that is five times faster, so a fixed per-iteration communication cost is a far larger
share of XCALibre's 174 ms iteration than of OpenFOAM's 526 ms one at the same subdomain size of
62,438 cells. That cannot be the whole story at a 29-point gap, so a real excess remains to be
found; splitting it needs `-log_view` at two and eight ranks.

Clock readings are not comparable between the two codes here: the OpenFOAM sample is taken after
the run exits while XCALibre's is taken inside the timed run, so only within-code trends mean
anything.

Final residuals again agree across all four rank counts to sixteen significant figures.

### Context: earlier unbound OpenFOAM runs

`log.PCG_diagonal_PBiCGStab_diagonal*` in the benchmark case, 2026-08-14, gave 100 / 83 / 76 / 69%
at 1 / 2 / 4 / 8 ranks from TOTAL `ExecutionTime` over 500 iterations, with no explicit binding.
Both differences matter: total time includes a tail where the pressure solve gets cheaper as the
case converges, which is why the same run is 3.03 s/iter over iterations 3 to 100 but 1.70 s/iter
averaged over 500. Kept for context, not comparable with the table above.
