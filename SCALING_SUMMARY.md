# Distributed (MPI) strong-scaling summary

Backward-facing step, laminar, incompressible, Float64 on CPU. All timings are
`(t100 - t3) / 97` s per SIMPLE iteration, which cancels JIT and setup cost. One MPI rank per
physical P-core, verified from each rank's kernel affinity mask.

Data: [`dev/telemetry/scaling.csv`](dev/telemetry/scaling.csv),
[`machine_bandwidth.csv`](dev/telemetry/machine_bandwidth.csv),
[`petsc_events.csv`](dev/telemetry/petsc_events.csv).
Plots: `dev/telemetry/plots/` — regenerate with `julia dev/scripts/plot_scaling.jl`.

---

## 1. The headline: measure with the clock pinned, or measure nothing

This machine (i9-14900HX, 8 P-cores) drops from 4400 MHz on one busy core to 3100 MHz on eight.
Rank count and clock therefore co-vary, and an uncorrected strong-scaling curve measures the
laptop's power limit rather than the code.

| 499,503 cells | n=2 | n=4 | n=6 | n=8 |
|---|---:|---:|---:|---:|
| clock free (what we first measured) | 87% | 73% | - | 47% |
| **clock pinned at 2200 MHz** | **102%** | **94%** | **83%** | **71%** |
| constant package power (cross-check) | 106% | 98% | - | 67% |

Two independent controls agree: pinning the clock in hardware (`no_turbo=1`), and loading every
otherwise-idle P-core with spin loops so all rank counts throttle equally. The commands for both
are in `dev/gotchas.md`.

![throttling](dev/telemetry/plots/throttling.png)

## 2. Efficiency is independent of mesh size

| pinned at 2200 MHz | n=2 | n=4 | n=6 | n=8 |
|---|---:|---:|---:|---:|
| 499,503 cells | 101.6% | 94.5% | 82.6% | 71.0% |
| 1,320,368 cells | 101.4% | 93.3% | 78.1% | 69.0% |

A 2.6x larger mesh reproduces the curve to within a few points. This rules out two explanations
at once: a communication-bound loss would **improve** with the better volume-to-surface ratio,
and a DRAM-bandwidth-bound loss would **worsen** with a working set of 32 MB instead of 12 MB.
Neither happens, so the loss is a fixed fraction of the work.

## 3. Is there a regression from n=6 to n=8? No.

Cumulative efficiency always falls, which makes 83% -> 71% look like a cliff. The marginal cost
of each added rank tells the real story:

| incremental efficiency | 1->2 | 2->4 | 4->6 | 6->8 |
|---|---:|---:|---:|---:|
| 499,503 cells | 102% | 93% | 87% | **86%** |
| 1,320,368 cells | 101% | 92% | 84% | **88%** |

The 6->8 step costs the same as the 4->6 step, on both meshes. There is no step change at full
core occupancy: the degradation is gradual and monotone from n=4 onward, and the n=8 point is
not anomalous.

## 4. Where the remaining loss is

From PETSc `-log_view` at 499,503 cells:

| event | n=2 | n=8 | rank imbalance at n=8 |
|---|---:|---:|---:|
| `KSPSolve` | 6.90 s | 5.12 s | 2.1x |
| `VecNorm` | 0.14 s (2% of solve) | **2.73 s (53% of solve)** | **38.2x** |
| `MatMult` | 4.77 s | 2.93 s | 1.8x |

Excluding `VecNorm`, the solve scales at 71% rather than 34% from two ranks to eight. The 38x
imbalance is *arrival spread*: ranks reach the all-reduce at different times and wait. METIS
balances owned cells to 1.001 but leaves ghost counts at 3.63x, so ranks do unequal halo work.
Changing the METIS objective does not fix it (volume cuts total ghosts 11% and leaves imbalance
at 3.66 against 3.63).

## 5. The machine's own ceiling

MPI `axpy`, the same kernel as PETSc's `VecAXPY`, aggregate GB/s:

| working set | n=1 | n=2 | n=4 | n=8 |
|---|---:|---:|---:|---:|
| cache resident | 72.7 | 110.3 | 375.4 | 989.0 |
| DRAM resident | 33.1 | 36.5 | 36.7 | **38.7** |

**DRAM bandwidth does not scale on this machine.** One core nearly saturates it; eight cores buy
17%. Cache bandwidth scales 13.6x. DDR5-5600 dual channel is 89.6 GB/s theoretical.

Placing PETSc against that ceiling:

- `VecAXPY` reaches 104.6 GB/s at n=2 — above the DRAM peak, so it never leaves cache — against
  a measured cache ceiling of 110.3. **PETSc runs at 95% of what the hardware allows.**
- `MatMult` streams a 54 MB matrix that cannot fit the 36 MB L3, at 15.8 to 25.7 GB/s against the
  flat 38.7 GB/s ceiling. This one term genuinely is DRAM-bound.

![machine ceiling](dev/telemetry/plots/machine_ceiling.png)

## 6. Against OpenFOAM, same mesh, same machine, both pinned at 2200 MHz

| n | XCALibre Cg+Jacobi | OpenFOAM GAMG | XCALibre eff | OF eff |
|---|---:|---:|---:|---:|
| 1 | 1.0665 | 1.0620 | 100% | 100% |
| 2 | 0.5251 | 0.5567 | 102% | 95% |
| 4 | 0.2822 | 0.3354 | 94% | 79% |
| 6 | 0.2152 | 0.2511 | 83% | 71% |
| 8 | 0.1877 | 0.2127 | 71% | 62% |

XCALibre matches OpenFOAM's production multigrid per iteration **using only Jacobi**, and scales
better at every rank count.

Running OpenFOAM with the *same* algorithm XCALibre uses (PCG + diagonal):

| n | OF PCG+diagonal | OF efficiency | XCALibre is |
|---|---:|---:|---:|
| 1 | 6.7125 | 100% | 6.29x faster |
| 2 | 3.1138 | 108% | 5.93x faster |
| 4 | 1.3201 | 127% | 4.68x faster |
| 8 | 0.6181 | **136%** | 3.29x faster |

**A slow baseline manufactures high efficiency.** OF PCG+diagonal scales superlinearly only
because its one-rank case is 6.3x slower and badly memory-starved, so extra ranks bring cache
that the baseline never had. Efficiency percentages are comparable only between configurations
of similar absolute speed — comparing a fast code's efficiency against a slow one's is
meaningless.

![efficiency](dev/telemetry/plots/efficiency.png)

## 7. Remedies tested

| change | n=8 s/iter | verdict |
|---|---:|---|
| Cg + Jacobi (baseline) | 0.1877 | - |
| Cg + BoomerAMG (hypre) | 0.2744 | **46% slower**, scaling no flatter (70.0% vs 71.0%) |
| PipeCG + Jacobi | 0.2200 | **17% slower** |

Neither remedy helps on this case.

**BoomerAMG** is slower at every rank count (1.5359 vs 1.0665 s/iter at n=1) because 500k cells
of laminar BFS is small and well-conditioned enough that the setup never pays for itself. Its
incremental 6->8 efficiency is better (93% against 86%), consistent with fewer Krylov iterations
meaning fewer reductions, but not nearly enough to overcome the per-iteration cost. AMG should be
revisited on a stiffer problem, not dismissed.

**Pipelined CG** overlaps the reduction with computation via non-blocking `MPI_Iallreduce`. It
took effect (`KSP=pipecg` confirmed in the solver log) and cost 17%: the extra work per iteration
needed to enable the overlap exceeds the barrier wait it hides at this scale.

![absolute cost](dev/telemetry/plots/absolute_cost.png)

## 8. Correctness

Residuals are identical across every rank count to 15 significant figures on both meshes, in
every configuration tested. Partitioning changes nothing about the answer.

## 9. What this means in practice

- Scaling to four ranks is essentially free (93-94%); to eight it costs about 30%.
- Prefer fewer, larger subdomains. Each Krylov iteration ends in a global reduction, and small
  subdomains spend a growing share of each iteration waiting at it.
- Do not exceed one rank per physical core.
- Expect a workstation or node with a sustained clock to beat these figures; a laptop cannot hold
  its clock under full load.
- The open optimisation is the reduction count, not bandwidth and not halo exchange: the Krylov
  solves issue roughly 75 reductions per outer iteration.

## 10. Corrections to earlier findings

Three claims made earlier in this work were wrong and are withdrawn:

1. **"The ceiling is memory bandwidth" (D16, withdrawn by D19).** Inferred from `VecAXPY`
   scaling at 63% without ever checking the rate against the machine's peak. It runs entirely in
   cache at 95% of the achievable rate.
2. **"XCALibre scales 29 points worse than OpenFOAM" (withdrawn by D22).** The OpenFOAM clocks
   were sampled *after* `foamRun` exited, so they read idle cores, and the comparison set
   XCALibre's Cg+Jacobi against OF's PCG+diagonal rather than its production GAMG. Corrected,
   XCALibre scales *better* at every rank count.
3. **"Communication-volume imbalance is the proximate cause" (D15, withdrawn by D16).**

Decisions D19-D24 in `dev/decisions.md` record each correction and its evidence.
