# P1-M27 - wall-function empty patches, KOmegaSST and potential_flow! distributed

Machine: 14 GB box, stock binaries, `dev/petscenv_stock`. Recorded 2026-09-20.

## Which rank counts empty which patch (`dev/scripts/patch_probe.jl`, 10 mm backward-facing step)

| ranks | per-rank face counts | empty wall-function patches |
|---|---|---|
| 4 | wall 23/22/23/42, top 22/23/22/33 | none (inlet and outlet only) |
| 6 | wall 20/12/34/15/14/15, top 32/23/0/15/16/14 | rank 2 `top` |
| 8 | wall 11/11/0/23/22/20/12/11, top 11/11/23/0/10/23/11/11 | rank 2 `wall`, rank 3 `top` |

The reported motorBike failure is the same shape: 353k cells, rank 2 `motorBike`=0 and rank 5 `lowerWall`=0 at n=6; rank 0 `lowerWall`=0 and ranks 6,7 `motorBike`=0 at n=8.

## Verdict runs

| run | ranks | result | wall time |
|---|---|---|---|
| `unit_test_wall_function_empty_patch.jl` (serial) | 1 | 4/4 | 3.9 s |
| `test_potential_flow.jl` (serial) | 1 | 501/501 | 6.8 s |
| `test_turbulence_sst_wallfn.jl` + `test_potential_flow_mpi.jl` | 2, 6 | 4/4 | 3m01s |
| `test_turbulence_sst_wallfn.jl` + `test_potential_flow_mpi.jl` | 8 | 2/2 | 2m14s |
| `test_potential_flow_3d_mpi.jl` (periodic cascade, `pref` live) | 2, 4 | 2/2 | 52 s |

## What isolated D147

`test_potential_flow_mpi.jl` at n=2 before the fix, against the serial reference:

| quantity | max owned-cell error |
|---|---|
| Phi | 1.08e-15 (rank 0), 4.86e-16 (rank 1) |
| corrected face flux `phif` | 1.67e-16 (rank 0), 9.71e-17 (rank 1) |
| U_x | 0.699 (rank 0), 0.251 (rank 1) |

The worst U cell on rank 0 touched no processor cut, which ruled out halo and ghost causes and pointed at `reconstruct!` alone.

## Not measured here

motorBike itself does not fit this machine: n=8 unguarded drove MemAvailable to zero and OOM-killed the editor (D144); n=6 under `memguard.sh` was killed at 1852 MB during JIT compilation, before the first iteration (D145). The per-case precompile recipe (D135) is the lever if it is wanted locally; otherwise it is a P2 run.
