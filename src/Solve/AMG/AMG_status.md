# AMG Status

## Summary

Stage B is implemented in the native AMG path, but the single-thread
cylinder target is still not met. The main blocker is now clear:
current AMG configurations do not reduce the pressure linear residual
fast enough per unit cost to beat the baseline `Cg()+Jacobi()` solve.

The earlier claim that the baseline was also often limited by `itmax`
was wrong for the current benchmark configuration and has now been
explicitly disproved.

## Implemented Work

The following Stage B work is in place:

- `RugeStuben()` now uses a two-pass coarse/fine split and
  operator-dependent direct interpolation.
- `SmoothAggregation()` now supports stronger aggregation and
  `near_nullspace=...`.
- New smoothers were added:
  - `AMGSymmetricGaussSeidel()`
  - `AMGL1Jacobi()`
- `AMG(...)` now supports `cycle=:V` and `cycle=:W`.
- Bounded hierarchy numeric reuse is implemented through:
  - `coarse_refresh_interval`
  - `numeric_refresh_rtol`
- Hierarchy diagnostics now track:
  - operator complexity
  - grid complexity
  - effective cycle convergence factor
- Adaptive rebuild is defensive:
  - a weak last cycle forces rebuild on the next `update!`.

The following correctness/performance fixes are also in place:

- Fixed the CG workspace aliasing bug where the solution shared storage
  with the `q = A*p` workspace.
- Fixed coarse stopping so coarsening stops before dropping below
  `max_coarse_rows`.
- Reduced avoidable hot-path allocations:
  - `_cpu_vector(::Vector)` returns the vector directly
  - coarse solve tries `ldiv!` before allocating `solver \ b`
- Fixed repeated hierarchy diagnostics so they are only logged on the
  initial hierarchy build.
- Corrected the cylinder benchmark harness so Stage B modes no longer
  use the stale misleading `itmax=1` setup.
- Added linear solve history instrumentation for both Krylov and AMG
  paths so comparisons can be made on matched linear residual behavior,
  not only on outer CFD residuals.

## Verified Findings

Verified on 2026-04-23 for the warmed single-thread CPU 5-step cylinder
run:

- Baseline `Cg()+Jacobi()` did not hit `itmax=1000` on any of the
  10 recorded pressure solves.
- Baseline pressure iteration counts ranged from `240` to `900`.
- Baseline final linear residuals were consistently near `1e-5`.
- `AMG(mode=:cg)` with the current SGS Stage B setup hit `itmax=40` on
  `7/10` pressure solves.
- `AMG(mode=:solver)` with the current SGS Stage B setup hit `itmax=40`
  on `10/10` pressure solves.

This is the key conclusion from the latest verification:

- the remaining AMG gap is not explained by the baseline also
  saturating its iteration cap
- the current AMG issue is structural/algorithmic, not just benchmark
  bookkeeping or allocation noise

## Latest Measured Results

Measured on 2026-04-23 with `JULIA_NUM_THREADS=1`:

| mode | elapsed_s | outer pressure residual | linear solve behavior |
| --- | ---: | ---: | --- |
| baseline `Cg()+Jacobi()` | `2.324711` | `p_last = 1.936286e-4` | `10/10` pressure solves converged before `itmax` |
| `AMG(mode=:cg)` + SGS, `cycle=:V`, `itmax=40` | `6.106369` | `p_last = 1.706483e-3` | `7/10` pressure solves hit `itmax` |
| `AMG(mode=:solver)` + SGS, `cycle=:W`, `itmax=40` | `14.818648` | `p_last = 1.063786e-1` | `10/10` pressure solves hit `itmax` |

Interpretation:

- baseline remains clearly better on single-thread CPU wall-clock
- AMG-CG is the strongest native AMG path so far, but it is still too
  expensive and still often iteration-limited
- direct AMG is materially weaker on this case and is still not
  competitive

## Solver Semantics

Current stopping logic is believed to be correct:

- `amg_solve!` and `amg_cg_solve!` stop when either:
  - absolute residual is `<= atol`
  - relative residual is `<= rtol`
  - iteration count reaches `itmax`
- `itmax` should be treated as a safety cap, not as a tuning knob to
  make AMG appear faster
- fair comparisons must use matched `atol` / `rtol` and recorded linear
  residual histories

## What Was Tested

Validated in this session:

- `test/test_AMG.jl`
- linear solve history capture for both Krylov and AMG paths
- warmed single-thread cylinder benchmark runs for:
  - baseline
  - `AMG(mode=:cg)` Stage B SGS configuration
  - `AMG(mode=:solver)` Stage B SGS configuration

Additional note:

- `test/test_smoothers.jl` was not used as a clean standalone signal for
  this session because the direct ad hoc invocation missed imports that
  are present in the normal test harness

## Mistakes To Avoid

- Do not assume CG instability implies a nonsymmetric operator; inspect
  the assembled matrix.
- Do not reuse one workspace vector for both the solution and Krylov
  temporaries.
- Do not use low `itmax` values to make AMG timing look better.
- Do not claim AMG is faster than the baseline without matched linear
  residual histories.
- Do not assume the bottleneck is mainly allocations without measuring;
  the current problem is primarily cost per useful residual reduction.

## Next Step

The next step should focus on reducing cost per useful residual
reduction on the single-thread CPU path, using the new instrumentation
as the guardrail.

Priority order:

1. Check code for bugs or algorithm mistakes (use reputable sources only)
2. Keep `itmax` generous so it is not the active limiter in fair
   comparisons.
3. Split timings into:
   - hierarchy build
   - hierarchy refresh
   - cycle / iteration apply cost
4. Reduce hierarchy and cycle cost without losing too much convergence:
   - lower operator complexity
   - reduce interpolation density where possible
   - avoid unnecessary coarse numeric refreshes
5. Re-check whether each change improves achieved linear residual at
   matched wall-clock against baseline.

Longer term, Stage C is still required:

- move AMG apply paths to backend-resident kernels
- make CUDA AMG backend-resident instead of CPU-backed
- replace repeated sparse rebuilds with symbolic-pattern reuse where
  possible
