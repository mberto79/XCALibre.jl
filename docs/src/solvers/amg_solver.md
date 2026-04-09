# AMG Solver

XCALibre.jl includes a built-in Algebraic Multigrid (AMG) linear solver that plugs
into the standard `SolverSetup` API alongside the existing Krylov solvers. It has no
external dependencies beyond those already in `Project.toml`.

## Quick start

```julia
solvers = (
    U = SolverSetup(solver = Bicgstab(), preconditioner = Jacobi(), ...),
    p = SolverSetup(
        solver         = AMG(
                            smoother      = JacobiSmoother(2, 2/3, zeros(0)),
                            cycle         = VCycle(),      # or WCycle()
                            coarsening    = :SA,           # or :RS
                            max_levels    = 20,
                            coarsest_size = 50,
                            pre_sweeps    = 2,
                            post_sweeps   = 2,
                            strength      = 0.25,
                         ),
        preconditioner = Jacobi(),   # ignored by AMG; kept for API compatibility
        convergence    = 1e-8,
        relax          = 0.2,
        rtol           = 1e-3,
        itmax          = 20,
    ),
)
```

`AMG` is a subtype of `AbstractLinearSolver`; `VCycle`, `WCycle`, and `Chebyshev`
are exported from `XCALibre.Solve`.

## Architecture

### File layout

```
src/Solve/AMG/
  AMG.jl              plain include list (no module wrapper — matches Preconditioners/)
  AMG_0_types.jl      VCycle/WCycle, Chebyshev, AMG marker, MultigridLevel, AMGWorkspace
  AMG_1_kernels.jl    @kernel SpMV, axpy/axpby, copy, zero, Dinv, Jacobi sweep
  AMG_2_coarsen.jl    Smoothed-Aggregation (:SA) and Ruge–Stüben (:RS) coarsening
  AMG_3_galerkin.jl   Tentative P, smoothed P, R = Pᵀ, Ac = RAP (SpGEMM), spectral radius
  AMG_4_smoothers.jl  Damped-Jacobi and Chebyshev level smoothers
  AMG_5_cycle.jl      V-cycle, W-cycle (recursive); cycle dispatch
  AMG_6_api.jl        _workspace, amg_setup!, update!, solve_system! dispatch, CSR helpers
```

### Type hierarchy

```
AbstractLinearSolver
└── AMG{S<:AbstractSmoother, C<:AMGCycle}    # user-facing zero-field marker

AbstractSmoother
├── JacobiSmoother                           # existing smoother; reused per level
└── Chebyshev{F}                             # polynomial smoother (new)

AMGCycle
├── VCycle
└── WCycle

MultigridLevel{Tv, AType, Vec}   # mutable; one instance per level in hierarchy
AMGWorkspace{Vec, Opts}          # mutable; stored in phiEqn.solver
```

`AMGWorkspace` exposes a `.x` field so the existing `_copy!` path in `solve_system!`
continues to work without modification.

## Solution process

### Phase 1 — Setup (once per sparsity pattern change)

`amg_setup!` builds the full coarse hierarchy on CPU, then transfers only what is
needed to the device:

1. **Finest level**: Wrap `A_device` directly (no matrix copy). Build `Dinv` via
   `amg_build_Dinv!` (`@kernel`, on device). Estimate spectral radius ρ(D⁻¹A)
   via 10 power-iteration steps on CPU.
2. **Coarsening loop** (repeated until `n ≤ coarsest_size` or `max_levels`):
   a. Aggregate nodes: `:SA` — MIS-style strength-of-connection graph; `:RS` —
      C/F split by greedy lambda scoring.
   b. Tentative prolongation `P̂` (one `1.0` per row, column = aggregate index).
   c. Smoothed prolongation: `P = P̂ − ω·D⁻¹·A·P̂`, with `ω = 4/(3ρ)`.
   d. Restriction: `R = Pᵀ`.
   e. Galerkin coarse matrix: `Ac = R·A·P` via two SpGEMM passes on CPU.
   f. Transfer `P`, `R`, `Ac` to device; cache CPU copies of `P`/`R` in the
      level struct to avoid round-trips on future `update!` calls.
   g. Allocate coarse work vectors (`x`, `b`, `r`, `tmp`, `Dinv`) on device.
3. **Coarsest level**: Dense LU factorisation on CPU host via `LinearAlgebra.lu!`.

### Phase 2 — Numerical update (once per outer SIMPLE/PISO iteration)

`update!` reuses the existing hierarchy (same sparsity pattern, new coefficients):

1. Rebuild `Dinv` for the finest level on device (one `@kernel` call).
2. For each inter-level transition: recompute `Ac = R·A·P` numerically using the
   **cached CPU copies** of `P` and `R` — no device→CPU copy of operators. Write
   updated `nzval` back to device via `_copy_nzval_to_device!`. Rebuild `Dinv` for
   the coarse level on device.
3. Re-factorize the coarsest dense LU.

### Phase 3 — V-cycle (every multigrid iteration)

All operations in the cycle are `@kernel` launches — no Julia allocations, no
host/device transfers:

```
vcycle!(levels, lvl):
    if lvl == coarsest:
        copyto!(lu_rhs, b)          # coarsest b → host (small, ≤ coarsest_size)
        ldiv!(lu_factor, lu_rhs)    # dense LU on host
        copyto!(x, lu_rhs)          # scatter solution back to device
        return
    pre-smooth (Jacobi/Chebyshev)   # @kernel on device
    r = b − A·x                     # amg_residual! (@kernel)
    b_c = R·r                       # amg_spmv! (@kernel)
    zero!(x_c)                      # @kernel
    vcycle!(levels, lvl+1)          # recurse
    x += P·x_c                      # amg_spmv_add! (@kernel)
    post-smooth                     # @kernel on device
```

W-cycle applies two recursive coarse solves between the two prolongation steps.

## Kernel strategy

Every hot-path operation is a `@kernel` macro launched via the existing
`_setup(backend, workgroup, ndrange)` idiom used throughout XCALibre.jl. This makes
all operations backend-agnostic: CPU (multi-threaded via KernelAbstractions.jl's
`CPU()` backend), CUDA, AMDGPU, and oneAPI all use the same code paths.

| Kernel | Operation |
|---|---|
| `_amg_spmv!` | `y = A·x` (row-per-workitem CSR) |
| `_amg_spmv_add!` | `y += α·A·x` |
| `_amg_axpy!` | `y += α·x` |
| `_amg_axpby!` | `y = α·x + β·y` |
| `_amg_copy!` | `dst = src` |
| `_amg_zero!` | `v .= 0` |
| `_amg_build_Dinv!` | `Dinv[i] = 1/A[i,i]` |
| `_amg_jacobi_sweep!` | one damped-Jacobi sweep (uses `tmp` as double-buffer) |

The Jacobi smoother uses a pointer swap (`level.x, level.tmp = level.tmp, level.x`)
between sweeps — zero allocation per sweep.

Coarsening and Galerkin products run on CPU (they occur only at setup, which is
amortised over all solver iterations). The `L2` norm delegates to
`LinearAlgebra.norm` which uses multi-threaded BLAS on CPU.

## Integration with XCALibre.jl

The dispatch hook is:

```julia
function solve_system!(
    phiEqn::ModelEquation{T,M,E,S<:AMGWorkspace,P}, setup, result, component, config
)
```

This takes precedence over the generic Krylov path by Julia's dispatch specificity
rules. The `solve_equation!` → `solve_system!` pipeline in `Solve_1_api.jl` is
**unchanged**. The `_workspace(::AMG, b)` constructor returns a lazy `AMGWorkspace`
(empty levels); the first call to `update!` inside `solve_system!` triggers the full
`amg_setup!`. Subsequent calls perform only the cheap numerical refresh.

## Memory layout

All work vectors (`x`, `b`, `r`, `tmp`, `Dinv`) are allocated once per level on the
target device during setup. The solve phase (V/W-cycle loop) performs zero host
allocations. The only device↔host transfers during a solve are at the coarsest level
(`copyto!` of a ≤50-element vector), which is unavoidable since the direct solve
uses CPU-side dense LU.
