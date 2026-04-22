# AMG Status

## Current implementation

The AMG framework lives under `src/Solve/AMG/` and is integrated into `Solve` through:

- `_workspace(::AMG, b)` for solver-owned workspace allocation
- `solve_system!(..., setup::SolverSetup{...,<:AMG,...}, ...)` for AMG dispatch
- `update!(workspace, A, solver, config)` for hierarchy creation/reuse

Public solver-side API currently exposed:

- `AMG`
- `SmoothAggregation`
- `RugeStuben`
- `AMGJacobi`
- `AMGChebyshev`

Supported solve modes:

- `mode=:solver`: AMG used directly as the pressure solver
- `mode=:cg`: AMG used as a preconditioner for CG

Hierarchy behavior:

- hierarchy is built once and cached in the AMG workspace
- cached hierarchy is reused when the matrix pattern is unchanged
- on reuse, only cheap finest-level updates are refreshed (`diag`, `invdiag`, spectral estimate)
- CPU hierarchy build uses CSR -> CSC conversion for coarse operators
- `SmoothAggregation` now applies a Jacobi-smoothed tentative prolongator instead of plain injection
- CUDA path currently builds and applies the AMG hierarchy on CPU using a CPU copy of the device matrix

Logging:

- hierarchy construction now emits an `@info` message
- it reports `mode`, number of `levels`, row-count chain, coarsening type, smoother type, and backend

## Important limitations

- The current implementation is functional for the validated smoke-test paths, but it is still a pragmatic v1.
- The `SmoothAggregation` path is still simplified relative to production AMG packages such as `AlgebraicMultigrid.jl`, PyAMG, or hypre:
  - aggregation is still a naive pairwise-first-neighbor strategy
  - tentative prolongation is still a piecewise-constant injection from those aggregates
  - there is no explicit near-nullspace candidate fitting/improvement step beyond the implicit constant candidate encoded by aggregation
- The level smoother choices are still limited to weighted Jacobi and Chebyshev. There is no symmetric Gauss-Seidel or l1-Jacobi style smoother yet.
- CUDA AMG is currently a CPU-backed fallback for hierarchy build and apply. The rest of the flow solve still runs on `CUDABackend()`, but AMG itself is not backend-resident on CUDA yet.
- Coarse matrices are still formed through fresh sparse products during hierarchy setup. This is acceptable now because hierarchy setup is cached, but it is still an area to revisit if setup must become cheaper.
- `AMGJacobi` has **not** been independently validated as a smoother beyond smoke/performance runs. This should be treated as not fully tested.
- For the transient cylinder pressure system, direct AMG as a solver is still not competitive with the default `Cg()+Jacobi()` configuration on CPU. The hierarchy quality improved, but not enough.

## What was tested

Tested:

- focused AMG unit test file `test/test_AMG.jl`
- CPU smoke tests for the cylinder example
- CUDA smoke test for the cylinder example using the CPU-backed AMG fallback
- full 50-iteration CPU run for the cylinder example
- full 50-iteration CUDA run for the cylinder example
- post-smoothed-prolongation focused AMG unit test run
- post-smoothed-prolongation CPU comparison against the baseline cylinder pressure solver
- post-smoothed-prolongation CPU run of `examples/2D_cylinder_U_AMG.jl` for 180 seconds

Not fully tested:

- `AMGJacobi` as a standalone smoother implementation beyond those runs
- broader SIMPLE/PISO cases beyond the cylinder configuration used here
- performance/general behavior of `RugeStuben` on large transient cases
- true GPU-resident AMG setup/apply
- full long-run CPU validation of the new smoothed-prolongation `SmoothAggregation` path

## Performance work completed

The main performance problem at the start of this session was that the AMG pressure path appeared to hang in PISO. The following steps were taken to improve performance:

1. Cached the hierarchy in the AMG workspace instead of rebuilding it on every pressure solve.
2. Added pattern checks so rebuild only happens when matrix size/nnz changes.
3. Reduced reuse-time work to cheap finest-level refreshes (`diag`, `invdiag`, `lambda_max`).
4. Added a Jacobi-smoothed tentative prolongator for `SmoothAggregation` instead of using the tentative prolongator directly.
5. Added a safe CPU-backed CUDA fallback so the GPU path runs instead of failing on scalar indexing.
6. Avoided rebuilding the CPU hierarchy during CUDA solve application by reusing the cached CPU hierarchy.
7. Cached the coarsest-level factorization so the coarse solve no longer refactorizes on every V-cycle.
8. Switched the example/performance path to `AMGJacobi()` instead of `AMGChebyshev()` for lower apply cost.
9. Set the example pressure AMG to `mode=:solver` with `itmax=1`, i.e. one V-cycle per pressure correction, which is appropriate for the transient PISO example.
10. Added bounded inner AMG iteration defaults by making `SolverSetup(..., solver=AMG(...))` default to `itmax=200`.
11. Exposed `smoothing_steps` in `AMG(...)` and defaulted it to 10, while preserving explicit `presweeps`/`postsweeps` overrides.

One attempted change was reverted:

- Threading the inner AMG vector loops with `Threads.@threads` caused very large allocation growth and worse runtime. That change was removed.

## Performance observations

Useful observations from this session:

- The dominant repeated cost was not hierarchy rebuild anymore after caching; it was repeated work inside AMG apply, especially the coarse solve before factorization caching.
- A one-V-cycle AMG pressure correction is much cheaper and is enough to keep the cylinder example within the target runtime.
- Enabling Jacobi prolongation smoothing changed the hierarchy chain on the cylinder matrix from
  `64468 -> 32784 -> 16613 -> 8453 -> 4356 -> 2301 -> 1263 -> 736 -> 468`
  to
  `64468 -> 32784 -> 16512 -> 8335 -> 4228 -> 2172 -> 1138 -> 619 -> 357`.
- The new smoothed-prolongation hierarchy is better behaved than the old unsmoothed one, but direct AMG is still much weaker than the default CPU pressure solve on the cylinder case.
- On matched short CPU runs of the cylinder case, the baseline pressure residuals remained much smaller:
  - baseline `Cg()+Jacobi()`: `9.93e-5, 1.61e-4, 2.23e-4`
  - direct `AMG(mode=:solver, ..., itmax=1)`: `3.81e-1, 5.31e-1, 5.24e-1`
  - `AMG(mode=:cg, ...)` still started much larger than baseline (`1.53e-3` at the first reported outer iteration) and was too slow to finish a 3-iteration comparison within 180 s.
- A 180 s CPU run of `examples/2D_cylinder_U_AMG.jl` with the new smoothed prolongator reached about 30% of the configured 500 iterations. It did not immediately explode, and the pressure residual decreased from about `5.31e-1` to about `7.75e-2`, but this is still far worse and slower than the default baseline path.
- For the cylinder example, the currently validated fast configuration is still:
  - `mode=:solver`
  - `coarsening=SmoothAggregation()`
  - `smoother=AMGJacobi()`
  - `smoothing_steps=10`
  - `max_levels=8`
  - `max_coarse_rows=100`
  - `itmax=1`

## Last validated example state

`examples/2D_cylinder_U_AMG.jl` is currently configured with:

- `iterations=500`
- default PISO `inner_loops=2`
- pressure solver:
  - `AMG(mode=:solver, coarsening=SmoothAggregation(), smoother=AMGJacobi(), smoothing_steps=10, max_levels=8, max_coarse_rows=100)`
  - `itmax=1`

Latest observed outcomes:

- A CPU run with the old unsmoothed hierarchy completed previously.
- After enabling smoothed prolongation, the CPU example did not complete within a 180 s timeout, but it progressed to about 30% of the configured run without the immediate large-growth behavior seen in some earlier direct-AMG tests.
- The updated example is therefore **not yet revalidated as a full replacement** for the default CPU pressure configuration.

## Suggested next steps

- Properly validate `AMGJacobi` with targeted convergence tests, not just smoke/performance runs.
- Add tests for cached hierarchy reuse and cached coarse-factor reuse.
- Replace the current pairwise-first-neighbor aggregation with a more standard aggregation strategy closer to `AlgebraicMultigrid.jl` / PyAMG `StandardAggregation`.
- Add explicit near-nullspace candidate handling and fitting for `SmoothAggregation`, rather than relying only on aggregate injection.
- Add a stronger CPU smoother for AMG levels, preferably symmetric Gauss-Seidel or l1-Jacobi.
- Re-evaluate `AMG(mode=:cg)` after the hierarchy and smoother quality improve; this is the more plausible path for matching the default CPU pressure solver on the cylinder case.
- Replace the CPU-backed CUDA fallback with a true backend-resident AMG path.
- Revisit preallocation for hierarchy setup if setup cost becomes important again.
- If setup cost matters, preallocate and update coarse operators in place rather than recreating them through fresh sparse products.
