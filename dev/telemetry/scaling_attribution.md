# Strong-scaling attribution, backward-facing step, Float64 CPU, Cg + Jacobi

Supersedes the memory-bandwidth attribution in `scaling_bfs_cg_jacobi.md` (D16, withdrawn by D19).

## Method

`(t100 - t3) / 97` cancels JIT and setup. Ranks bound one per physical P-core, verified from
each rank's kernel affinity mask (`0,1`, `2,3`, ... = one core with both siblings), so eight
ranks use eight distinct cores. Clock pinned for the definitive runs with
`no_turbo=1`, `min_perf_pct=100`, governor `performance`; the clock actually held was sampled
during each run, not after it.

## The machine's own ceiling (MPI axpy, same kernel as VecAXPY)

| working set | n=1 | n=2 | n=4 | n=8 |
|---|---:|---:|---:|---:|
| DRAM-resident, 480 MB/vector | 33.1 | 36.5 | 36.7 | 38.7 GB/s |
| cache-resident, 0.5-4 MB/vector | 72.7 | 110.3 | 375.4 | 989.0 GB/s |

DRAM bandwidth is flat: one core nearly saturates it, eight cores buy 17%. Cache bandwidth
scales 13.6x. DDR5-5600 dual channel is 89.6 GB/s theoretical.

## Where PETSc sits against that ceiling (499,503 cells, -log_view)

- `VecAXPY` 104.6 GB/s at n=2 against a 110.3 GB/s cache ceiling: 95% of achievable, and above
  the DRAM peak, so it never leaves cache. Not bandwidth-limited. This refutes D16.
- `MatMult` 15.8 to 25.7 GB/s streaming a 54 MB matrix that cannot fit the 36 MB L3, against a
  flat 38.7 GB/s ceiling. This one term is genuinely DRAM-bound.
- `VecNorm` 53% of the n=8 solve against 2% at n=2, 38.2x rank imbalance. Excluding it the
  solve scales at 71% rather than 34%. This is the dominant residual (D23).

## Efficiency, clock pinned at 2200 MHz (D20, D21)

| n | 499,503 cells | 1,320,368 cells |
|---|---:|---:|
| 2 | 102% | 101% |
| 4 | 94% | 93% |
| 6 | - | 78% |
| 8 | 71% | - |

Unpinned the same cases read 87/73/47% and 85/68/52%, with cores falling 4400 to 3100 MHz as
rank count rises. A 2.6x larger mesh changes efficiency by one point, so the loss is a fixed
fraction of the work: not communication, which would improve, and not DRAM bandwidth, which
would worsen.

Cross-check at constant package power (spin loops on every P-core the solver is not using,
so all rank counts throttle equally): 106% at n=2, 98% at n=4, 67% at n=8. Agrees.

Residuals are identical across every rank count to 15 significant figures on both meshes.

## Against OpenFOAM, same mesh, same machine, both pinned at 2200 MHz (D22)

| n | XCALibre Cg+Jacobi | OpenFOAM GAMG | XCALibre eff | OF eff |
|---|---:|---:|---:|---:|
| 1 | 1.0665 | 1.0620 | 100% | 100% |
| 2 | 0.5251 | 0.5567 | 102% | 95% |
| 4 | 0.2822 | 0.3354 | 94% | 79% |
| 8 | 0.1877 | 0.2127 | 71% | 62% |

XCALibre matches OpenFOAM's production multigrid using only Jacobi, and scales better at every
rank count. OF GAMG reads 0.5297 s/iter unpinned against 1.0620 pinned, exactly 2.00x for half
the clock, which validates the pin.
