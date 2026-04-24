# AMG Status

This file is the compact working status for the native AMG solver.
Detailed design/review context lives in `src/Solve/AMG/AMG_plan.md`.

Update rules:

- replace snapshot numbers and tables instead of appending new benchmark diaries
- keep only durable conclusions, current blockers, and next actions
- add at most one short line per material change in the decision log
- keep raw benchmark dumps, residual traces, and exploratory notes out of this file

Historical detail from the previous long-form status was condensed here on
2026-04-23 so the file can stay useful during continued AMG iteration.

## Current Snapshot

Last reviewed: 2026-04-24

Status:

- Stage B feature work is largely in place in the native AMG path.
- `SmoothAggregation()` now uses a matching-seeded standard aggregate
  builder instead of the earlier greedy seed/fill pass.
- `SmoothAggregation()` now defaults to sparse filtered smoothing with a
  capped interpolation stencil (`filter_weak_connections=true`,
  `max_interp_entries=4`) so the default AMG-CG path avoids the
  operator-complexity blow-up seen with dense smoothed prolongation.
- A two-pass sparse-SA interpolation variant was benchmarked as the
  next coarse-correction improvement candidate.
- The strongest native path on the cylinder benchmark is
  `AMG(mode=:cg)` used as a light reusable preconditioner.
- The single-thread CPU target is still not met:
  `Cg()+Jacobi()` remains materially faster and reaches a lower pressure
  residual.
- Direct AMG (`AMG(mode=:solver)`) is not currently competitive on this
  benchmark.

Benchmark conditions for the numbers below:

- warmed 5-step cylinder benchmark
- `JULIA_NUM_THREADS=1`
- matched linear solve tolerances with recorded linear residual history

## Current Best Verified Results

Measured on 2026-04-24 after benchmarking the sparse-SA pass-count
variant:

| path | elapsed_s | outer result | linear solve behavior | verdict |
| --- | ---: | ---: | --- | --- |
| baseline `Cg()+Jacobi()` | `2.113334` | `p_last = 1.936286e-4` | `10/10` pressure solves converged before `itmax` | current wall-clock and convergence reference |
| sparse SA `pass1`, `max_interp_entries=4` | `6.240556` | `p_last = 1.825867e-3` | `9/10` pressure solves hit `itmax` | current sparse-SA performance baseline |
| sparse SA `pass2`, `max_interp_entries=4` | `7.592050` | `p_last = 1.624881e-3` | `8/10` pressure solves hit `itmax` | slightly better coarse correction, but slower overall |
| exploratory symmetric-strength SA | `11.449468` | `p_last = 1.756935e-3` | `10/10` pressure solves hit `itmax` in the recorded run | not promising; coarsening stalls and wall-clock worsens |

Latest timing split:

| path | build/rebuild_s | refresh_s | finest_refresh_s | apply_s |
| --- | ---: | ---: | ---: | ---: |
| sparse SA `pass1`, `max_interp_entries=4` | `0.742771` | `0.176105` | `0.009138` | `4.999291` |
| sparse SA `pass2`, `max_interp_entries=4` | `1.045412` | `0.334511` | `0.009084` | `5.904578` |

## Accepted Conclusions

- The earlier idea that the baseline was often limited by `itmax` was
  false for the current benchmark configuration.
- After the rebuild-policy fix, the remaining AMG-CG gap is mostly
  weak residual reduction per apply, not hierarchy maintenance
  overhead.
- Lighter `V(1,1)` AMG-CG cycles are better than the earlier heavier
  `V(2,2)` stock path on this benchmark.
- Dense smoothed prolongation is not viable with the new aggregate
  builder; sparse filtering and interpolation caps need to stay in the
  default SA path.
- The sparse-default SA change fixes the operator-complexity blow-up,
  but it does not materially improve residual reduction per AMG-CG
  iteration.
- The two-pass sparse-SA interpolation experiment is not a performance
  win on the cylinder case: it lowers the final pressure residual and
  reduces `itmax` hits slightly, but it also raises operator complexity
  and apply cost enough to lose on wall-clock.
- The remaining near-term opportunity is still coarse-correction
  quality, but only if the added reach can be combined with stronger
  sparsification than the current two-pass variant.
- Symmetric-strength SA is still not a good next path on this benchmark:
  it coarsens too slowly and loses on wall-clock.
- Direct AMG should not be the next optimisation priority for this
  benchmark; the better near-term path is still a cheaper AMG-CG
  preconditioner.

## Implemented Capabilities And Fixes

Current native AMG work that should be treated as present baseline
functionality:

- `RugeStuben()` uses a two-pass coarse/fine split and
  operator-dependent direct interpolation.
- `SmoothAggregation()` supports:
  - `near_nullspace`
  - `truncate_factor`
  - `max_interp_entries`
  - `interpolation_passes`
  - `strength_measure`
  - `filter_weak_connections`
- `SmoothAggregation()` now builds aggregates with a matching-seeded
  standard aggregation pass and only falls back to singletons for
  unavoidable isolated leftovers.
- The default `SmoothAggregation()` policy is now sparse:
  - `filter_weak_connections=true`
  - `max_interp_entries=4`
  - `interpolation_passes=2` in code, but this is not yet the
    recommended maintained benchmark path
- Smoothers include:
  - `AMGSymmetricGaussSeidel()`
  - `AMGL1Jacobi()`
  - `AMGChebyshev()`
  - `AMGJacobi()`
- `AMG(...)` supports `cycle=:V` and `cycle=:W`.
- Fixed-pattern reuse is the default via
  `assume_fixed_pattern=true`, with checked structural reuse still
  available through `assume_fixed_pattern=false`.
- Adaptive rebuild behavior is mode-aware:
  - `AMG(mode=:solver)` can rebuild after weak reused AMG-cycle behavior
  - `AMG(mode=:cg)` does not rebuild from overall Krylov convergence
- Numeric reuse controls are implemented through:
  - `coarse_refresh_interval`
  - `numeric_refresh_rtol`
- Diagnostics now track:
  - operator complexity
  - grid complexity
  - effective cycle convergence factor
- Instrumentation now records:
  - matched linear residual histories
  - hierarchy build/rebuild time
  - hierarchy refresh time
  - finest-only refresh time
  - AMG apply time
- AMG solve reporting now distinguishes a true `itmax` exit from a
  solve that reaches tolerance on its final allowed iteration.
- Correctness/performance fixes already landed include:
  - CG workspace aliasing fix
  - coarse stopping fix for `max_coarse_rows`
  - reduced hot-path allocations
  - hierarchy diagnostics logged only on initial build
  - defensive rebuild only after weak reused hierarchies

## Validation Coverage

- `test/test_AMG.jl` covers:
  - standard aggregation pairing and leftover handling
  - reuse vs rebuild update behavior
  - fixed-pattern semantics
  - truncation behavior
  - V-cycle and W-cycle operation
  - smoother behavior
  - AMG-CG rebuild semantics
- The latest benchmark review validated behavior for:
  - baseline `Cg()+Jacobi()`
  - sparse SA `pass1`, `max_interp_entries=4`
  - sparse SA `pass2`, `max_interp_entries=4`
  - symmetric-strength SA
- The current session validated the interpolation-pass plumbing and
  updated AMG semantics through the focused AMG test slice in
  `test/test_AMG.jl`.

## Guardrails

- Keep `itmax` generous; do not use it to make AMG look faster.
- Compare solver variants only at matched `atol`/`rtol` with recorded
  linear residual histories.
- Treat cost per useful residual reduction as the primary metric, not
  allocations alone.
- Do not interpret weaker Krylov convergence in `AMG(mode=:cg)` as
  automatic proof that the hierarchy is stale; lighter preconditioners
  can still be the right tradeoff.
- On this benchmark, prefer cheaper V-cycle preconditioning over
  restoring heavier cycles unless coarsening quality clearly improves.

## Next Work

- Keep AMG-CG on the reusable `V(1,1)` path as the maintained baseline.
- Keep the sparse-default SA policy in place; do not regress to dense
  smoothed prolongation.
- Do not treat the current two-pass sparse-SA interpolation variant as a
  performance improvement; benchmark says it is slower despite slightly
  better coarse correction.
- Recommended next step:
  - either revert the maintained benchmark path to the one-pass sparse
    SA variant
  - or keep the two-pass idea only as an experimental branch and add a
    stronger post-pass sparsification/filter so the extra reach does not
    drive operator complexity and apply cost up
- Candidate directions after that:
  - filtered long-range interpolation with explicit secondary
    sparsification and a hard row cap preserved after the long-range
    pass
  - coarse-level sparsification targeted at reducing `A_c` complexity
  - HMIS/PMIS-like splitting only if sparse-SA coarse correction still
    remains too weak after the cheaper long-range path is exhausted
- Re-check every change against the baseline at matched wall-clock and
  matched linear residual behavior.
- Stage C remains necessary for backend-resident apply paths and
  genuinely GPU-native AMG.

## Compact Decision Log

| date | decision / finding |
| --- | --- |
| 2026-04-23 | Fixed-pattern reuse became the default for the current fixed-mesh CFD use case; checked structural reuse remains available as an opt-out path. |
| 2026-04-23 | AMG-CG rebuild decisions should not be driven by overall Krylov convergence; rebuild behavior is now mode-aware. |
| 2026-04-23 | The lighter `V(1,1)` AMG-CG configuration replaced the older heavier Stage B path as the working baseline. |
| 2026-04-23 | Sparse filtered SA with `max_interp_entries=4` became the default because it removes the operator-complexity blow-up while preserving the previous best AMG-CG behavior. |
| 2026-04-24 | The two-pass sparse-SA interpolation experiment improved residual reduction slightly but lost on wall-clock because operator complexity and AMG apply cost increased too much. |
| 2026-04-23 | Direct AMG rebuild pathology was reduced, but direct AMG still loses mainly on apply cost and weak per-cycle convergence. |
| 2026-04-23 | SmoothAggregation switched to a matching-seeded standard aggregate builder; dense smoothed prolongation is too expensive without sparse filtering on this benchmark. |
| 2026-04-23 | Symmetric-strength SA still coarsens too slowly on the cylinder case and is not the next performance path. |
