# Add AMG Solver Framework To `Solve`

## Summary
Implement a backend-resident AMG solver stack under `src/Solve/AMG/` that plugs into the existing `SolverSetup` and `solve_system!` flow without changing the current user-facing configuration style. The new solver will support:
- Pure AMG solve mode.
- Hybrid AMG-preconditioned CG mode for SPD systems such as pressure.
- Two coarsening families: `SmoothAggregation()` and `RugeStuben()`.
- Two GPU-friendly smoothers: `AMGJacobi()` and `AMGChebyshev()`.
- In-place `update!` of the hierarchy against the current matrix values to reduce allocations inside SIMPLE/PISO loops.

Competitive v1 requirements are based on common production AMG features exposed by [hypre BoomerAMG](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html), [PETSc GAMG](https://petsc.org/release/manualpages/PC/PCGAMG/), and [NVIDIA AmgX](https://developer.nvidia.com/amgx/): reusable hierarchy setup, Jacobi/Chebyshev smoothing, AMG-as-solver and AMG-with-Krylov usage.

## Key Changes
### Public API
Add new solver/coarsening/smoother types re-exported from `XCALibre.Solve`:
- `AMG <: AbstractLinearSolver`
- `SmoothAggregation`
- `RugeStuben`
- `AMGJacobi`
- `AMGChebyshev`

Use solver-level configuration, not preconditioner-level configuration:
```julia
p = SolverSetup(
    solver = AMG(
        mode = :cg,                 # :solver or :cg
        coarsening = SmoothAggregation(),
        smoother = AMGChebyshev(),
        presweeps = 1,
        postsweeps = 1,
        max_levels = 10,
        min_coarse_rows = 32,
        max_coarse_rows = 256
    ),
    preconditioner = Jacobi(),      # ignored by AMG path; kept for API compatibility
    convergence = 1e-7,
    relax = 1.0,
    rtol = 0.0,
    atol = 1e-5
)
```
Behavior:
- `mode=:solver` runs an AMG V-cycle as the main solver with early exit from `atol`/`rtol`.
- `mode=:cg` runs CG with AMG V-cycle as the preconditioner; only valid for symmetric matrices.
- Existing Krylov solvers remain unchanged.
- `preconditioner` stays mandatory in `SolverSetup` for compatibility, but AMG code ignores it and uses its own hierarchy/smoother stack.

### Internal Architecture
Create `src/Solve/AMG/` with XCALibre naming:
- `AMG.jl`
- `0_AMG_types.jl`
- `1_AMG_setup.jl`
- `2_AMG_coarsening.jl`
- `3_AMG_transfer.jl`
- `4_AMG_smoothers.jl`
- `5_AMG_cycle.jl`
- `6_AMG_cg.jl`
- `7_AMG_update.jl`

Integrate by:
- Including `AMG/AMG.jl` from `src/Solve/Solve.jl`.
- Extending `_workspace(::AMG, b)` to allocate an AMG workspace analogous to current Krylov workspaces.
- Extending `solve_system!` with an AMG dispatch that:
  - reads `itmax`, `atol`, `rtol`,
  - calls `update!(workspace, A, config)` before the solve,
  - runs either V-cycle iterations or AMG-CG,
  - writes the solution back into the field values like the current path.

### Data Structures And Backend Rules
Represent the hierarchy as backend-local level structs:
- matrix `Aℓ`
- optional strong-connection graph data
- prolongation `Pℓ`
- restriction `Rℓ`
- diagonal / inverse diagonal / Chebyshev coefficients
- residual, correction, temporary vectors
- coarse solve scratch
- metadata: `nrows`, level id, smoother config

Implementation rules:
- All vectors and sparse matrices for a level live on the active backend.
- All setup/apply kernels use `KernelAbstractions`.
- Matrix-value-only refreshes reuse existing sparsity and update diagonals, smoother coefficients, and Galerkin coarse matrices in place.
- Pattern-changing rebuilds trigger a hierarchy rebuild; v1 assumes XCALibre’s matrix pattern is fixed across nonlinear iterations and only values change.

### Algorithm Choices
For `SmoothAggregation()`:
- Build backend-local aggregate ids from a strength graph.
- Form tentative prolongator by piecewise-constant aggregates.
- Smooth prolongator with damped Jacobi using an estimated spectral radius.
- Use `R = transpose(P)` and Galerkin `A_c = R * A * P`.

For `RugeStuben()`:
- Build strong connections from a threshold on off-diagonal strength.
- Compute C/F splitting.
- Build classical interpolation from strong coarse neighbors.
- Use Galerkin coarse operators.

Smoothers:
- `AMGJacobi`: weighted Jacobi, fully fused KA kernels where possible.
- `AMGChebyshev`: degree-2 or degree-3 polynomial smoother using diagonal scaling and estimated eigenvalue bounds from a few power iterations / Gershgorin-style lower bound; no triangular solves.

Coarse level:
- Stop coarsening at `max_coarse_rows` or `min_coarse_rows`.
- CPU backend: direct dense/CSR solve on the coarsest level.
- CUDA backend: copy only the final coarse matrix/vector to CPU for the coarsest solve in v1, then return the correction to device. All higher levels remain backend-local.

Deferred from v1:
- aggressive coarsening,
- block/coupled AMG,
- nonsymmetric Krylov hybrids beyond CG,
- distributed-memory coarse-grid repartitioning.

### Example, Docs, And Validation Surface
Add:
- one focused AMG example derived from [examples/2D_cylinder_U.jl](/home/humberto/Julia/XCALibre.jl/examples/2D_cylinder_U.jl) with pressure using `AMG(mode=:cg, coarsening=SmoothAggregation(), smoother=AMGChebyshev())`;
- user-guide updates for the new solver options;
- release note entry.

The first implementation step should create `src/AMG_plan.md` and store this plan there verbatim, because this turn is still in Plan Mode and does not modify the repo.

## Test Plan
Add automated tests in the existing suite for:
- API construction: `SolverSetup(solver=AMG(...))` on CPU and, when CUDA is available, CUDA.
- Hierarchy setup: SA and RS both produce at least one coarse level and backend-local work arrays.
- `update!`: changing only `nzval` updates diagonals/coarse operators without reallocating level counts or pattern metadata.
- Smoothers: `AMGJacobi` and `AMGChebyshev` reduce residual on a Laplacian SPD system on CPU; same on CUDA when available.
- Pure AMG solve: converges on a manufactured 2D Laplace system and respects `atol` / `rtol` early exit.
- AMG-CG solve: converges on the same SPD system and matches or improves iteration count versus plain `Cg()`+`Jacobi()`.
- Coarsening coverage: one test for `SmoothAggregation()`, one for `RugeStuben()`.
- Failure mode: `AMG(mode=:cg)` throws a clear error if the caller uses it on a matrix marked or detected as nonsymmetric.
- End-to-end example: add a reduced-iteration cylinder regression derived from `2D_cylinder_U.jl` and require it to run without errors on CPU; add a CUDA-gated version for environments with CUDA.

Acceptance for implementation:
- `Pkg.test()` passes on CPU.
- The new AMG cylinder example runs without errors on CPU.
- If CUDA is available locally, the same example’s pressure solve path runs without errors with `CUDABackend()`.

## Assumptions And Defaults
- Full backend-resident hierarchy setup is required for supported backends; only the coarsest-grid direct solve may fall back to CPU in v1.
- Initial supported GPU acceptance target is CUDA; AMDGPU/oneAPI support follows the same KA path but is not milestone-blocking.
- `AMG` is solver-owned, not exposed as a `preconditioner` type in v1.
- `atol` and `rtol` are checked every outer AMG iteration or every CG iteration using the existing residual definition.
- XCALibre’s sparsity pattern is assumed stable through nonlinear iterations, so `update!` is value-update-first and rebuild-on-pattern-change.
- Pressure is the main initial target; momentum equations continue to use the current Krylov solvers unless users opt into AMG manually.
