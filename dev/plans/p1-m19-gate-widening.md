# P1-M19 - gate widening (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R8, R9, Q1, Q2. Governing decisions: D72, D73.

## Problem, quantified

The gate is five files at n=2 (`test/distributed/gate.jl`); ghost consistency after each self-syncing primitive is asserted by nothing; agreement across rank counts (15 significant figures under Jacobi, `dev/telemetry/scaling_attribution.md`) is a manual check; a redundant exchange added by mistake fails no test. Odd rank counts, which give asymmetric neighbour lists and a rank that may own zero faces of a patch, are never run.

## Approach

Widen the existing driver rather than add a framework. The ghost check is a debug helper in `Distribute` that any future physics model reuses.

## Configuration space

ranks {1, 2, 3, 5, 8 oversubscribed} x case {BFS 10 mm, cavity} x precond {Jacobi}; `runtests_mpi.jl --ranks=...` is the table.

## Steps

- [ ] **P1-M19-S1** `gate.jl` runs at ranks `[2, 3]`; `runtests_mpi.jl` defaults to `[1, 2, 3, 5]`; a documented `--ranks=8` invocation with `--oversubscribe` (MPICH: `-launcher fork`, verify the stock launcher's flag) is run once and recorded - mechanism: odd counts exercise asymmetric halos - cost: gate time roughly doubles - verdict: gate green and within Q2 (300 s) at a pinned 2200 MHz; if over, drop `test_partition.jl` from the n=3 leg and record the time.
- [ ] **P1-M19-S2** `Distribute.check_ghosts(x, dm, config)` exchanges into a scratch copy and returns the max absolute ghost mismatch; `test_ghosts.jl` runs one SIMPLE iteration on BFS at n=2,3 with the check after `grad!`, `limit_gradient!`, `inverse_diagonal!`, `H!`, `explicit_relaxation!`, each `solve_system!`, `wall_distance!` and SST `nut`, asserting exactly zero - mechanism: ghost equals owner is the invariant every sync! exists to hold - cost: one exchange per check, test-only - verdict: zero mismatch on every primitive; any nonzero names the primitive that lost its sync.
- [ ] **P1-M19-S3** `test_invariance.jl`: BFS psimple, Jacobi, 50 iterations, residual histories at n=1,2,4 agree to a relative 1e-12 per iteration - mechanism: Jacobi is partition-invariant and reductions are the only order change - cost: three runs - verdict: green; record the observed spread in `gate_results.md`.
- [ ] **P1-M19-S4** `HaloExchange` and `residual` carry `Ref{Int}` counters; `test_perf.jl` asserts exchanges and all-reduces per SIMPLE iteration against a recorded budget (today: 8 exchanges, 8 all-reduces for laminar SIMPLE, `AUDIT.md`) - mechanism: a new round is a regression unless a decision lowers the budget - cost: two integer increments - verdict: counts equal the budget; P1-M22 lowers it with a decision.

## Exit criterion

Gate green at n=2 and n=3 within Q2; `test_ghosts.jl`, `test_invariance.jl` and the counters green at n=1,2,3,5; the n=8 oversubscribed run recorded.

## Open questions

- Whether `test_partition.jl` at n=3 or n=5 exposes a patch with zero local faces on the 10 mm BFS; if not, add the cavity with a 3-way split.
