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
        relax          = 1.0,
        rtol           = 1e-3,       # exit early when residual drops to 0.1% of initial
        atol           = 1e-5,
        itmax          = 10,         # maximum V-cycles per solve call
    ),
)
```

`AMG` is a subtype of `AbstractLinearSolver`; `VCycle`, `WCycle`, and `Chebyshev`
are exported from `XCALibre.Solve`.

### Convergence within each solve call

After each V-cycle, the solver checks:

```
res_norm < atol  ||  res_norm / r0 < rtol
```

and exits early if either condition is met. A well-built hierarchy typically satisfies
`rtol = 1e-3` in 1–3 cycles for a pressure Laplacian, so `itmax = 5–10` is usually
sufficient.

## Architecture

### File layout

```
src/Solve/AMG/
  AMG.jl              plain include list (no module wrapper — matches Preconditioners/)
  AMG_0_types.jl      VCycle/WCycle, Chebyshev, AMG marker, GalerkinPlan,
                      LevelExtras, MultigridLevel, AMGWorkspace
  AMG_1_kernels.jl    @kernel SpMV, axpy/axpby, copy, zero, Dinv, Jacobi sweep,
                      fused Galerkin kernel (_amg_galerkin!)
  AMG_2_coarsen.jl    Smoothed-Aggregation (:SA) and Ruge–Stüben (:RS) coarsening
  AMG_3_galerkin.jl   Tentative P, smoothed P, R = Pᵀ, Ac = RAP (SpGEMM),
                      _build_galerkin_plan, spectral radius estimation
  AMG_4_smoothers.jl  Damped-Jacobi and Chebyshev level smoothers
  AMG_5_cycle.jl      V-cycle, W-cycle (recursive); cycle dispatch
  AMG_6_api.jl        _workspace, amg_setup!, update!, solve_system! dispatch,
                      CSR helpers, _fill_dense_from_sparse!
```

### Type hierarchy

```
AbstractLinearSolver
└── AMG{S<:AbstractSmoother, C<:AMGCycle}    # user-facing marker type

AbstractSmoother
├── JacobiSmoother                           # damped Jacobi (default)
└── Chebyshev{F}                             # polynomial smoother

AMGCycle
├── VCycle
└── WCycle

GalerkinPlan{Vi<:AbstractVector{Int32}}     # device-resident pre-computed plan
                                            # for the fused R·A·P KA kernel

LevelExtras{Tv, CpuSpT}                    # host-only mutable per-level state
  ├── P_cpu, R_cpu  : CPU copies of P and R (for plan rebuild on amg_setup! re-call)
  ├── A_cpu         : CPU copy of coarsest matrix (for LU rebuild in update!)
  ├── galerkin_plan : GalerkinPlan (device-resident); nil at coarsest level
  ├── lu_dense      : pre-allocated n×n dense buffer for in-place LU (coarsest only)
  ├── rho           : spectral radius of D⁻¹A (Chebyshev smoothing only)
  ├── lu_factor     : LU factorisation (coarsest only)
  └── lu_rhs        : scratch vector for LU back-solve

MultigridLevel{Tv, AType, PType, Vec, ExtrasT}  # one per level in hierarchy
AMGWorkspace{LType, Vec, Opts}                  # stored in phiEqn.solver
```

`AMGWorkspace` exposes a `.x` field so the existing `_copy!` path in `solve_system!`
continues to work without modification.

## Solution process

### Phase 1 — Setup (once per sparsity pattern change)

`amg_setup!` builds the full coarse hierarchy on CPU, constructs the device-resident
`GalerkinPlan` for each level, then transfers operators to the device.

1. **Finest level**: Wrap `A_device` directly (no matrix copy). Build `Dinv` on device
   via `amg_build_Dinv!`.

2. **Coarsening loop** (repeated until `n ≤ coarsest_size` or `max_levels`, stopping
   early if coarsening reduces the matrix by less than 10%):

   a. **Strength of connection**: `:SA` and `:RS` both use the standard min-max
      criterion — edge `(i,j)` is **strong** if
      `|a_ij| ≥ θ · max_{k≠i} |a_ik|`. This is appropriate for M-matrices (FVM
      pressure Laplacian) where diagonal ≫ individual off-diagonals. The previous
      symmetric criterion (`θ·√(|a_ii|·|a_jj|)`) was too strict for unstructured
      FVM meshes and caused coarsening to stall.

   b. **`:SA`** — Maximal Independent Set (MIS-2) aggregation on the strong-connection
      graph; `:RS` — C/F splitting via greedy lambda scoring.

   c. Tentative prolongation `P̂` (one `1.0` per row, column = aggregate index).

   d. Smoothed prolongation: `P = P̂ − ω·D⁻¹·A·P̂`.
      Damping `ω = 4/3` (analytic bound) when using `JacobiSmoother` — no power
      iteration needed. `ω = 4/(3ρ)` with power-iteration spectral radius estimate
      when using `Chebyshev`.

   e. Restriction: `R = Pᵀ`.

   f. Galerkin coarse matrix: `Ac = R·A·P` via two SpGEMM passes on CPU.

   g. **Build `GalerkinPlan`**: iterate over all `(i, k, l, j)` quadruples where
      `R[i,k] ≠ 0`, `A[k,l] ≠ 0`, `P[l,j] ≠ 0`, and record the triple of
      nonzero indices `(nzi_R, nzi_A, nzi_P)` for each output position `Ac[i,j]`.
      This CSR-structured plan is transferred to the device once and reused at
      every `update!` call.

   h. Transfer `P`, `R`, `Ac` to device. Store CPU copies of `P` and `R` in
      `LevelExtras` (for plan rebuild if `amg_setup!` is called again).

   i. Allocate coarse work vectors (`x`, `b`, `r`, `tmp`, `Dinv`) on device.

3. **Coarsest level**: Allocate a pre-allocated `n×n` dense buffer `lu_dense`.
   Fill from the sparse coarse matrix and factorise in-place via `LinearAlgebra.lu!`.
   The same buffer is refilled and re-factorised on every `update!` call (no
   allocation).

A diagnostic message prints the level sizes and whether direct solve is active:
```
[ Info: AMG hierarchy (SA, strength=0.25): [7234, 2410, 820, 275, 48] — direct solve at coarsest: true
```

### Phase 2 — Numerical update (once per outer SIMPLE/PISO iteration)

`update!` reuses the existing hierarchy (same sparsity pattern, new coefficients).
**No CPU↔device transfers occur except for the tiny coarsest-level nzval download.**

1. Rebuild `Dinv` for the finest level on device (one `@kernel` call).

2. For each level `i → i+1`, launch the **fused Galerkin kernel**:
   ```
   amg_galerkin!(Lc.A, L.A, L.R, L.P, plan, backend, workgroup)
   ```
   The kernel reads the current (device-resident) nzval arrays of A, R, and P
   using the pre-stored index triples from `GalerkinPlan`, and writes the new nzval
   of `Ac` directly — one KA thread per output nonzero. Then rebuilds `Dinv` for
   the coarse level.

3. Download the coarsest nzval from device to CPU (≤ `coarsest_size` entries ≈ 2 KB),
   refill the pre-allocated `lu_dense` buffer, and re-factorise in-place:
   ```julia
   copyto!(ex_c.A_cpu.nzval, nzval_device)   # ~2 KB, device → CPU
   fill!(ex_c.lu_dense, 0); _fill_dense_from_sparse!(ex_c.lu_dense, ex_c.A_cpu)
   ex_c.lu_factor = lu!(ex_c.lu_dense)        # in-place, no allocation
   ```

### Phase 3 — V-cycle (every multigrid iteration)

All operations in the cycle are `@kernel` launches — no Julia allocations, no
host/device transfers:

```
vcycle!(levels, lvl):
    if lvl == coarsest:
        copyto!(lu_rhs, b)          # b → CPU (≤ coarsest_size elements)
        ldiv!(lu_factor, lu_rhs)    # dense LU on CPU
        copyto!(x, lu_rhs)          # solution → device
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

Every hot-path operation is a `@kernel` macro launched via the
`_setup(backend, workgroup, ndrange)` idiom used throughout XCALibre.jl. All kernels
are backend-agnostic: CPU (multi-threaded via `CPU()` backend), CUDA, AMDGPU, and
oneAPI all use the same code paths.

| Kernel | Operation |
|---|---|
| `_amg_spmv!` | `y = A·x` (row-per-workitem CSR) |
| `_amg_spmv_add!` | `y += α·A·x` |
| `_amg_axpy!` | `y += α·x` |
| `_amg_axpby!` | `y = α·x + β·y` |
| `_amg_copy!` | `dst = src` |
| `_amg_zero!` | `v .= 0` |
| `_amg_build_Dinv!` | `Dinv[i] = 1/A[i,i]` |
| `_amg_dinv_axpy!` | `x[i] += ω·Dinv[i]·r[i]` (correction-form Jacobi update) |
| `_amg_galerkin!` | `Ac.nzval[out] = Σ R[nzi_R]·A[nzi_A]·P[nzi_P]` (fused R·A·P, one thread per output nonzero) |

The Jacobi smoother uses the **correction form** — compute `r = b − Ax`, then
`x += ω·D⁻¹·r` — rather than the classical two-buffer swap. This keeps
`MultigridLevel` immutable and avoids an extra device allocation per level.

Coarsening and Galerkin SpGEMM (symbolic) run on CPU at setup time (amortised over
all solver iterations). The fused Galerkin kernel (`_amg_galerkin!`) handles all
numerical updates on the device at runtime.

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

## Memory layout and data flow

```
                 CPU                        Device
──────────────────────────────────────────────────────────
Setup  P_cpu, R_cpu              P_dev, R_dev (fixed)
       A_cpu (coarsest only)     A_dev per level (changes)
       GalerkinPlan (CPU)   →→→  GalerkinPlan (device)
       lu_dense (dense n×n)      —
──────────────────────────────────────────────────────────
update!
 Galerkin   (nothing)       ←    A_dev (read; nzval changed by kernel)
            (nothing)       →→   Ac_dev (kernel writes new nzval)
 LU rebuild  A_cpu.nzval  ←←    Ac_dev.nzval  (≤50 entries, ~2 KB)
──────────────────────────────────────────────────────────
V-cycle     (nothing)       ←→   all work vectors on device
 coarsest   lu_rhs         ←←    b_c  (≤50 entries)
            lu_rhs         →→    x_c  (≤50 entries)
──────────────────────────────────────────────────────────
```

All work vectors (`x`, `b`, `r`, `tmp`, `Dinv`) are allocated once per level on the
target device during setup. The solve phase (V/W-cycle loop) performs **zero host
allocations and zero device↔host transfers** except at the coarsest level.

The `update!` path is similarly allocation-free: `GalerkinPlan` indices are fixed, the
fused kernel overwrites nzval arrays in-place, `lu_dense` is refilled without
allocation, and `lu!` factorises in-place (only the tiny `ipiv` Vector is allocated by
`LinearAlgebra.lu!`, ≈ 400 bytes for a 50×50 matrix).

## Performance notes

- **SA coarsening** uses the min-max strength criterion `|a_ij| ≥ θ·max_{k≠i}|a_ik|`.
  For an FVM pressure Laplacian on an unstructured 2D mesh, `strength = 0.25` gives
  good aggregation (most connections are strong relative to the row maximum).
  Typical reduction: 3–5× per level, reaching `coarsest_size = 50` in 4–6 levels.

- **Spectral radius** is skipped entirely when `smoother isa JacobiSmoother`
  (the default). This saves `10 × n_levels` sparse matrix-vector products at setup.

- **Coarsest-level LU** is rebuilt on every `update!` call via an in-place fill of a
  pre-allocated dense matrix. For `coarsest_size = 50` this is negligible (~2500
  float assignments + O(n³) = 125k FLOPs).

- **GPU**: the fused `_amg_galerkin!` kernel eliminates all fine-matrix CPU↔GPU
  transfers that the previous approach (CPU-side SpGEMM) required. The net result is
  that `update!` consists entirely of device-side kernel launches plus a ≈2 KB
  coarsest nzval download.
