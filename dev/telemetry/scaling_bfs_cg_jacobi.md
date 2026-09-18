> SUPERSEDED for the ATTRIBUTION by `scaling_attribution.md` (D19-D22). The memory-bandwidth
> conclusion here is withdrawn: the measurements were taken unpinned, so rank count and CPU
> clock co-varied, and `VecAXPY` was never checked against the machine's peak. The raw
> numbers below stand as recorded; only their interpretation is replaced.

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

## Where the loss is: PETSc stage split, bfs_tet_5mm, 2026-09-17

`petsc_options="-log_view"` through `run!`, 34 SIMPLE iterations (`KSPSolve` count 136 = 34 x 4
solves, three velocity components plus pressure), same binding as above.

| quantity | n=2 | n=8 | speedup on 4x the ranks |
|---|---:|---:|---:|
| total, s/iter | 0.456 | 0.2075 | 2.20x (55%) |
| `KSPSolve`, s/iter | 0.203 | 0.151 | 1.35x (34%) |
| everything else, s/iter | 0.253 | 0.057 | 4.45x (111%) |

**XCALibre's own discretisation, gradient, interpolation and halo work scales superlinearly.**
All of the parallel efficiency loss is inside the PETSc Krylov solve, whose share of an iteration
rises from 45% to 73% between two and eight ranks.

Inside the solve, the two events that grow are global synchronisation, not arithmetic:

| event | n=2 time | n=2 imbalance | n=8 time | n=8 imbalance |
|---|---:|---:|---:|---:|
| `VecNorm` | 0.136 s | 1.1 | 2.731 s | 38.2 |
| `VecScatterEnd` | 0.390 s | 7.5 | 1.386 s | 16.9 |
| `MatMult` | 4.773 s | 1.1 | 2.927 s | 1.8 |

`VecNorm` is an all-reduce: a 38x imbalance there is ranks ARRIVING at the reduction at different
times, not the reduction being slow. `VecScatterEnd` is the wait for halo data. Both point at the
same upstream cause, and PETSc names it directly: `MPI Msg Len` imbalance 3.566 at eight ranks.

### The partition is balanced by cell count and not by communication

Read back from the decompositions themselves:

| ranks | owned cells max/min | ghost cells max/min | halo as share of owned |
|---:|---:|---:|---:|
| 2 | 1.00 | 1.02 | 0.5% |
| 4 | 1.00 | 2.11 | 1.4% |
| 8 | 1.00 | 3.63 | 3.0% |

Cell counts are equal to three decimal places at every rank count, while one rank carries 3.63
times the halo of another at eight ranks: the 3.566 PETSc measured. Metis is being asked for
`:KWAY` with the default edge-cut objective, which balances vertices and minimises TOTAL cut
without balancing per-rank communication volume.

## The ceiling is memory bandwidth, not communication

`VecAXPY` is the control: it performs no communication at all, and PETSc reports it balanced
across ranks (imbalance 1.0 at two, 1.1 at eight). Its work per rank falls exactly fourfold from
two ranks to eight, from 2.27 to 0.568 Gflop.

| event | communication | n=2 time | n=8 time | speedup on 4x the ranks | n=8 imbalance |
|---|---|---:|---:|---:|---:|
| `VecAXPY` | none | 0.520 s | 0.207 s | 2.51x (63%) | 1.1 |
| `MatMult` | halo scatter | 4.773 s | 2.927 s | 1.63x (41%) | 1.8 |
| `VecNorm` | all-reduce | 0.136 s | 2.731 s | 0.05x | 38.2 |

An embarrassingly parallel, perfectly balanced, communication-free vector update achieves 63%.
That is the machine's memory bandwidth saturating as eight performance cores contend for one
DDR5 controller, and it is a ceiling on everything above it. `MatMult` sits below that ceiling
because it also carries the halo scatter, and `VecNorm` measures the arrival spread that
bandwidth jitter produces, which is why its imbalance is 38 while its own arithmetic is trivial.

### Partition objective does not move it

Repartitioning the same mesh under each Metis objective, counting ghost cells per rank:

| ranks | objective | ghost max/min | total ghosts |
|---:|---|---:|---:|
| 4 | cut (default) | 2.11 | 7066 |
| 4 | volume | 2.06 | 6216 |
| 4 | cut + minconn | 2.11 | 7066 |
| 8 | cut (default) | 3.63 | 15199 |
| 8 | volume | 3.66 | 13495 |
| 8 | cut + minconn | 3.63 | 15199 |
| 8 | cut + ufactor 1 | 3.77 | 15394 |

The volume objective cuts TOTAL communication by 11% and leaves the IMBALANCE untouched. The
imbalance is a property of the domain: a channel gives its end subdomains few neighbours and its
interior ones many, whatever the objective. It is also far too small to be the mechanism — 2373
against 654 ghost cells is about 19 kB against 5 kB an exchange, which cannot produce a
millisecond of all-reduce wait. It correlates with the loss; it does not cause it.

### Approximation in the split above

`-log_view` totals cover the whole process, which is warm-up plus the three-iteration and
thirty-iteration timed runs, while the probe's per-iteration figure covers the long run only.
The shares are therefore good to about five points; the 4.45x against 1.35x ratio that the
conclusion rests on is not sensitive to it.
