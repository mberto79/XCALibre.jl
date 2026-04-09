# AMG Solver — Implementation Plan & Design Reference

## Overview

This document describes the Algebraic Multigrid (AMG) linear solver added to
`XCALibre.jl` in `src/Solve/AMG/`.  It satisfies the requirements in
`src/AMG_instructions.md`:

- **Zero third-party dependencies** beyond those already in `Project.toml`
  (`KernelAbstractions`, `SparseMatricesCSR`, `Adapt`, `Atomix`, `LinearAlgebra`).
- **Hardware-agnostic** via `KernelAbstractions.jl` `@kernel` macros; GPU-specific
  wiring is unchanged from the existing extension pattern in `ext/`.
- **Integrated** with the existing `SolverSetup` / `solve_system!` API.
- **Updateable** via `update!(ws, A, backend, workgroup)` for cheap per-outer-iteration
  coefficient refreshes.

---

## File Layout

```
src/Solve/AMG/
  AMG.jl              aggregator (plain include list, no module wrapper)
  AMG_0_types.jl      VCycle, WCycle, Chebyshev, AMG marker, MultigridLevel, AMGWorkspace
  AMG_1_kernels.jl    @kernel SpMV, axpy, axpby, copy, zero, Dinv, Jacobi-sweep
  AMG_2_coarsen.jl    Smoothed-Aggregation (:SA) and Ruge–Stüben (:RS) coarsening
  AMG_3_galerkin.jl   Tentative P, smoothed P, R=Pᵀ, Ac=RAP (SpGEMM), spectral radius
  AMG_4_smoothers.jl  Damped-Jacobi and Chebyshev level smoothers
  AMG_5_cycle.jl      V-cycle, W-cycle (recursive); cycle dispatch
  AMG_6_api.jl        _workspace, amg_setup!, update!, solve_system! dispatch, CSR helpers
```

`src/Solve/Solve.jl` was edited to add `include("AMG/AMG.jl")` after `Solve_1_api.jl`.

---

## User-Facing API

```julia
using XCALibre

solvers = (
    U = SolverSetup(
        solver         = Bicgstab(),
        preconditioner = Jacobi(),
        convergence    = 1e-8,
        relax          = 0.8,
    ),
    p = SolverSetup(
        solver         = AMG(
                            smoother   = JacobiSmoother(; domain=mesh, loops=2, omega=2/3),
                            cycle      = VCycle(),       # or WCycle()
                            coarsening = :SA,            # or :RS
                            max_levels = 20,
                            pre_sweeps = 2,
                            post_sweeps = 2,
                            strength   = 0.25,
                         ),
        preconditioner = Jacobi(),   # retained for API compatibility; not used by AMG
        convergence    = 1e-8,
        relax          = 0.2,
        rtol           = 1e-3,
        itmax          = 20,
    ),
)
```

`AMG` subtype of `AbstractLinearSolver`; `Chebyshev` and `VCycle`/`WCycle` are
exported from `XCALibre.Solve`.

---

## Architecture

### Type hierarchy

```
AbstractLinearSolver
└── AMG{S<:AbstractSmoother, C<:AMGCycle}   # zero-field marker (like Bicgstab)

AbstractSmoother
├── JacobiSmoother    # existing; reused as AMG level smoother
└── Chebyshev         # new polynomial smoother

AMGCycle
├── VCycle
└── WCycle

MultigridLevel{Tv, AType, Vec}   # mutable; one per level
AMGWorkspace{Vec, Opts}          # mutable; stored in phiEqn.solver
```

### Integration contract

| XCALibre hook | AMG implementation |
|---|---|
| `_workspace(::AMG, b)` | Returns `AMGWorkspace` with empty levels (lazy build) |
| `phiEqn.solver` field | Holds `AMGWorkspace` (instead of Krylov workspace) |
| `solve_system!(phiEqn::ModelEquation{…,S<:AMGWorkspace,…}, …)` | Dispatches to `_solve_system_amg!` |
| `update!(ws, A, backend, workgroup)` | Refreshes coarse-level nzval; skips re-coarsening |

The `solve_equation!` → `solve_system!` pipeline in `Solve_1_api.jl` is
**unchanged**; dispatch is purely by type specialisation on `S<:AMGWorkspace`.

### Setup phase (`amg_setup!`)

1. Gather fine-level matrix `A_device` to CPU as `SparseMatrixCSR`.
2. Compute `Dinv_fine` (1/diagonal) and spectral radius ρ via 10 power-iteration steps.
3. Loop coarsening:
   a. Aggregate nodes (SA: strength-of-connection → MIS seed expansion; RS: C/F split).
   b. Build tentative prolongation `P̂` (one 1.0 per row, column = aggregate index).
   c. Smooth: `P = (I − ω_P D⁻¹ A) P̂` with `ω_P = 4/(3ρ)`.
   d. Restrict: `R = Pᵀ` (CSR transpose).
   e. Galerkin: `Ac = R A P` (two-pass SpGEMM on CPU).
   f. Transfer `P`, `R`, `Ac` to device.  Allocate coarse work vectors on device.
4. Build dense LU for the coarsest level (size ≤ `coarsest_size`).
5. Store hierarchy in `ws.levels`; set `ws.setup_valid = true`.

### Update phase (`update!`)

Reuses the existing hierarchy when the sparsity pattern is unchanged:
1. Rebuild `Dinv` for level 1 (coefficients changed in-place by outer solver).
2. For each coarse level: recompute `Ac = R A P` numerically via `galerkin_product`;
   write updated `nzval` into the device matrix (`_copy_nzval_to_device!`);
   rebuild `Dinv`.
3. Re-factorize the coarsest LU.

### Solve phase (V-cycle)

```
function vcycle!(levels, lvl, opts, backend, workgroup)
    if lvl == n_levels
        amg_coarse_solve!(levels[n])   # host LU → scatter to device
        return
    end
    pre-smooth!(L)                     # Jacobi or Chebyshev (device)
    r = b - A*x                        # amg_residual!
    b_c = R * r                        # amg_spmv! (coarsen)
    zero!(x_c)
    vcycle!(levels, lvl+1, ...)        # recurse
    x += P * x_c                       # amg_spmv_add! (prolong)
    post-smooth!(L)
end
```

W-cycle applies two recursive coarse solves between prolongation steps.

### Kernels (`AMG_1_kernels.jl`)

All in `@kernel` form, launched with `_setup(backend, workgroup, ndrange)`:

| Function | Description |
|---|---|
| `amg_spmv!` | `y = A*x` (row-per-workitem CSR) |
| `amg_spmv_add!` | `y += α*A*x` |
| `amg_axpy!` | `y += α*x` |
| `amg_axpby!` | `y = α*x + β*y` |
| `amg_copy!` | `dst = src` |
| `amg_zero!` | `v .= 0` |
| `amg_build_Dinv!` | `Dinv[i] = 1/A[i,i]` |
| `amg_jacobi_sweep!` | One damped-Jacobi sweep (uses `tmp` as swap buffer) |

All kernels work on CPU (via KA's `CPU()` backend with `Threads.@spawn` under the hood) and GPU backends unchanged.

---

## Verification

```bash
# 1. Build check
julia --project -e 'using XCALibre'

# 2. Unit test (Laplace-based Poisson problem)
julia --project test/unit_test_amg.jl   # (file to be created)

# 3. Integration: replace Krylov pressure solver with AMG in any SIMPLE example
#    Compare convergence history against Bicgstab+DILU baseline

# 4. Allocation check (inside V-cycle after warmup)
julia --project -e '
using XCALibre, BenchmarkTools
# ... build phiEqn, call solve once to trigger setup ...
@allocated solve_system!(phiEqn, setup, result, nothing, config)  # should be 0
'

# 5. GPU: CUDA.allowscalar(false) + run with AMG pressure solver
```
