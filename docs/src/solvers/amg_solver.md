# AMG Solver

XCALibre.jl includes a built-in Algebraic Multigrid (AMG) linear solver that plugs
into the standard `SolverSetup` API. It has no external dependencies beyond those
already in `Project.toml` and runs on CPU and all GPU backends supported by
KernelAbstractions.jl.

## How AMG works

AMG solves a sparse linear system `Ax = b` by building a hierarchy of progressively
coarser representations of the operator A, then using that hierarchy to eliminate
error components at all frequencies simultaneously. It operates in three distinct
phases — **setup**, **update**, and **solve** — and supports two solve modes.

### Phase 1 — Setup (hierarchy construction)

The setup phase runs once per sparsity pattern change (typically once per simulation,
since the mesh topology is fixed). It builds a sequence of coarser grids entirely on
the CPU:

```
Fine grid A₀  →  coarsen  →  A₁  →  coarsen  →  A₂  →  ···  →  Aₙ (direct solve)
               P₀, R₀=P₀ᵀ       P₁, R₁=P₁ᵀ
```

For each level, a **prolongation operator** `P` (fine ← coarse) and **restriction
operator** `R = Pᵀ` (fine → coarse) are constructed from the strength-of-connection
graph of A. The coarse matrix is the Galerkin product `Aₖ₊₁ = Rₖ Aₖ Pₖ`. The process
repeats until the matrix is small enough for a direct dense LU solve. All operators are
then transferred to the target device (GPU or CPU).

### Phase 2 — Update (numerical refresh)

Each outer SIMPLE/PISO iteration changes the coefficients of A but not its sparsity
pattern. The update phase recomputes all nzval arrays using a pre-built **fused
Galerkin kernel** that reads the new A.nzval on the device and writes each Aₖ.nzval
directly — no CPU↔device transfers for the hierarchy itself. The coarsest-level LU
is rebuilt from a tiny (≤ 2 KB) download of the coarsest nzval. The fine-level
diagonal D⁻¹ is always updated; coarse levels are refreshed every `update_freq` calls.

### Phase 3 — Solve: two modes

Each call to `solve_system!` executes the inner linear solve. The behaviour depends on
the `krylov` field of `AMG`.

---

#### Mode A — Richardson iteration (`krylov = :none`)

AMG is used as a **standalone iterative solver**. Each iteration applies one V-cycle
directly to the current iterate `x`:

```
x ← warm start from field values
r₀ ← b − A x
for k = 1 .. itmax:
    x ← V-cycle(x, b)        ← one multigrid cycle updating x in-place
    check ‖b − Ax‖ < atol or rtol · r₀
```

The **V-cycle** works top-down then bottom-up through the hierarchy:

```
vcycle at level ℓ:
    pre-smooth:    x ← x + ω D⁻¹(b − Ax)     (Jacobi/Chebyshev, pre_sweeps times)
    restrict:      b_{ℓ+1} ← R · (b − Ax)     (residual transferred to coarse grid)
                   x_{ℓ+1} ← 0
    coarse solve:  vcycle(ℓ+1)                 (recursive; direct LU at coarsest)
    prolongate:    x ← x + P · x_{ℓ+1}        (coarse correction added back)
    post-smooth:   x ← x + ω D⁻¹(b − Ax)     (post_sweeps times)
```

Each V-cycle damps error components across all length scales. For the FVM pressure
Laplacian, typical convergence factors are ρ ≈ 0.05–0.3 per cycle. Richardson is the
simplest path: lowest overhead per iteration, and competitive when the solve exits
early via `rtol`.

---

#### Mode B — Preconditioned CG (`krylov = :cg`, default)

The V-cycle is used as a **preconditioner** M⁻¹ ≈ A⁻¹ inside a Preconditioned
Conjugate Gradient (PCG) outer loop. Instead of applying V-cycles to the original
system directly, PCG builds a sequence of A-conjugate search directions that span the
Krylov subspace, choosing the optimal step size at each iteration:

```
x ← warm start from field values
r ← b − A x
z ← M⁻¹r          ← one V-cycle: RHS = r, initial guess = 0, output = z
p ← z;  ρ = rᵀz
for k = 1 .. itmax:
    Ap ← A p
    α  = ρ / (pᵀ Ap)         ← optimal step length
    x += α p                  ← update solution
    r -= α Ap                 ← update residual
    check ‖r‖ < atol or rtol · r₀
    z ← M⁻¹r                 ← one V-cycle: RHS = r, initial guess = 0, output = z
    β  = (rᵀz_new) / ρ       ← conjugation coefficient
    p  = z + β p              ← new A-conjugate search direction
    ρ  = rᵀz_new
```

Each PCG step costs one V-cycle (preconditioner apply) plus one SpMV (A·p) and two
dot products (global reductions). The key property is **Krylov optimality**: PCG
minimises the A-norm of the error over the entire Krylov subspace spanned so far,
reaching lower residuals per V-cycle than Richardson. This makes PCG the better
choice when the solve would otherwise consume all `itmax` cycles without converging.

> **Implementation note:** `vcycle!` overwrites the fine-level residual buffer `L1.r`
> with its own internal residual mid-cycle. After each preconditioner apply, the PCG
> residual `r` is restored from `L1.b` (the V-cycle RHS, which the cycle never
> modifies) before computing any dot products.

---

#### Choosing between the two modes

| Situation | Recommended |
|---|---|
| Tight `atol`, large `itmax`, inner convergence quality matters | `krylov = :cg` |
| Outer PISO/SIMPLE loop uses many pressure correctors | `krylov = :cg` |
| Large `rtol` (e.g., `1e-2`) — solve exits in 1–5 cycles | `krylov = :none` |
| GPU with many short solves where dot-product syncs dominate | `krylov = :none` |

---

## Quick start

```julia
solvers = (
    U = SolverSetup(solver = Bicgstab(), preconditioner = Jacobi(), ...),
    p = SolverSetup(
        solver = AMG(
            smoother      = JacobiSmoother(3, 2/3, zeros(0)),
            cycle         = VCycle(),
            coarsening    = :SA,
            max_levels    = 20,
            coarsest_size = 50,
            pre_sweeps    = 2,
            post_sweeps   = 1,
            strength      = 0.0,
            update_freq   = 2,
            krylov        = :cg,
        ),
        preconditioner = Jacobi(),   # required by API; ignored by AMG
        convergence    = 1e-7,
        relax          = 1.0,
        rtol           = 1e-3,
        atol           = 1e-5,
        itmax          = 20,
    ),
)
```

`AMG`, `VCycle`, `WCycle`, `JacobiSmoother`, and `Chebyshev` are exported from
`XCALibre.Solve`.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `smoother` | `JacobiSmoother(2, 2/3, zeros(0))` | `JacobiSmoother(sweeps, ω, _)` or `Chebyshev(degree, lo, hi)` |
| `cycle` | `VCycle()` | `VCycle()` or `WCycle()` (two coarse solves per level; harder problems) |
| `coarsening` | `:SA` | `:SA` Smoothed Aggregation or `:RS` Ruge–Stüben |
| `max_levels` | `25` | Maximum number of grid levels |
| `coarsest_size` | `50` | Stop coarsening when matrix dimension ≤ this; direct LU solve used |
| `pre_sweeps` | `2` | Smoothing sweeps before restriction |
| `post_sweeps` | `2` | Smoothing sweeps after prolongation |
| `strength` | `0.0` | Strength-of-connection threshold θ (see below) |
| `update_freq` | `1` | Refresh coarse hierarchy every N outer iterations |
| `krylov` | `:cg` | `:cg` (PCG, recommended) or `:none` (Richardson/V-cycle loop) |

### Strength-of-connection (`strength`)

An edge `(i,j)` is kept when `|a_ij| ≥ θ · max_{k≠i} |a_ik|`. **For FVM pressure
Laplacians the correct value is `strength = 0.0`** (keep all connections). These are
near-isotropic M-matrices where every off-diagonal entry carries equal weight; dropping
connections with `θ > 0` degrades prolongation quality and causes convergence failure
on non-uniform meshes. Use a non-zero θ only for strongly anisotropic problems where
selective coarsening along the dominant direction is intentional.

### Lazy hierarchy refresh (`update_freq`)

In a SIMPLE/PISO loop, `update!` is called once per outer iteration. With
`update_freq = N`, the fine-level diagonal D⁻¹ is always refreshed (cheap and
accuracy-critical), but the Galerkin products and coarsest-level LU are recomputed
only on calls 1, N+1, 2N+1, … This is safe because the outer loop is itself an
iterative correction; slightly stale coarse operators cost at most a few extra AMG
iterations per outer step. Recommended values: `1` (fully accurate), `2` (transient
with many time steps), `3–5` (near-converged steady runs).

### Krylov acceleration (`krylov`)

| `krylov` | Algorithm | Cost per step | Convergence |
|---|---|---|---|
| `:cg` (default) | Preconditioned CG; one V-cycle = preconditioner M⁻¹ | 1 V-cycle + 1 SpMV + 2 dots | Krylov-optimal; lower residual per V-cycle |
| `:none` | Richardson; each step is one V-cycle | 1 V-cycle | Linear; simpler |

PCG is Krylov-optimal: it reaches a lower inner residual for the same number of
V-cycles. Each step adds one SpMV (A·p) and two global reductions (dot products),
adding roughly 25–50% overhead per timestep. PCG recovers this cost when the solve
would otherwise hit `itmax` without meeting `atol` — the V-cycles are spent more
effectively. Richardson is faster when the solve exits early via `rtol` (large relative
tolerance or warm start).

**Guidance:**
- `krylov = :cg` — preferred when tight `atol` or large `itmax`; better inner
  pressure convergence reduces PISO/SIMPLE outer iterations.
- `krylov = :none` — preferred when throughput matters more than inner accuracy
  (e.g., large `rtol = 1e-2`, GPU with many short solves).

## Architecture

### File layout

```
src/Solve/AMG/
  AMG_0_types.jl   VCycle/WCycle, Chebyshev, AMG, GalerkinPlan,
                   LevelExtras, MultigridLevel, AMGWorkspace
  AMG_1_kernels.jl @kernel SpMV, axpy/axpby, copy, zero, Dinv (fast + slow path),
                   Jacobi sweep, fused Galerkin kernel
  AMG_2_coarsen.jl SA and RS coarsening, pairwise fallback
  AMG_3_galerkin.jl tentative P, smoothed P, R = Pᵀ, Ac = RAP (SpGEMM),
                   _build_galerkin_plan, Gershgorin spectral radius
  AMG_4_smoothers.jl damped Jacobi and Chebyshev level smoothers
  AMG_5_cycle.jl   V-cycle, W-cycle; cycle dispatch
  AMG_6_api.jl     _workspace, amg_setup!, update!, _amg_pcg_solve!,
                   solve_system! dispatch, CSR helpers
```

### Key types

```
AMGWorkspace{LType, Vec, Opts}        stored in phiEqn.solver
  ├── levels :: Vector{LType}         fully-typed: no dynamic dispatch in cycle
  ├── x, x_pcg, p_cg :: Vec          solution and PCG work vectors
  └── opts   :: AMG

MultigridLevel{Tv, AType, PType, Vec, ExtrasT}   one per grid level
  ├── A, P, R :: device-resident sparse matrices
  ├── x, b, r, tmp, Dinv :: device-resident work vectors
  └── extras :: LevelExtras           host-only mutable state

LevelExtras{Tv, CpuSpT}
  ├── P_cpu, R_cpu, A_cpu            CPU copies for setup/LU rebuild
  ├── galerkin_plan :: GalerkinPlan  device-resident plan for fused R·A·P kernel
  ├── lu_dense, lu_factor, lu_rhs   coarsest-level direct solve
  ├── rho                            spectral radius (Chebyshev only)
  └── diag_ptr :: AbstractVector{Int32}  branch-free diagonal lookup
```

`MultigridLevel` is immutable and carries `Adapt.@adapt_structure` for transparent
GPU migration. `LevelExtras` has no Adapt method and always stays on the host.

## Solution process

### Setup (`amg_setup!`, once per sparsity pattern)

Builds the full hierarchy on CPU, then transfers operators to device:

1. Compute strength-of-connection graph; run SA or RS coarsening to produce
   aggregate assignments.
2. Build tentative prolongation P̂ (piecewise constant), smooth it:
   `P = P̂ − ω D⁻¹ A P̂` where `ω = 4/(3ρ)` and ρ is the Gershgorin upper bound
   on ρ(D⁻¹A) (deterministic, O(nnz), exact for FVM M-matrices).
3. Build restriction `R = Pᵀ` and coarse matrix `Ac = R A P` via two SpGEMM passes.
4. Build `GalerkinPlan`: for each structural nonzero of Ac, record all
   `(nzi_R, nzi_A, nzi_P)` index triples such that
   `Ac[i,j] = Σ R.nzval[nzi_R] · A.nzval[nzi_A] · P.nzval[nzi_P]`.
   The plan is transferred to device once and reused at every `update!` call.
5. Coarsest level: fill a pre-allocated dense buffer and factorise in-place via
   `lu!` (guarded by `n ≤ coarsest_size`; falls back to 50 Jacobi sweeps otherwise).
6. Build `diag_ptr[i]` = nzval index of A[i,i] for each level; used for
   branch-free, divergence-free Dinv rebuilds on GPU.

### Numerical refresh (`update!`, once per outer SIMPLE/PISO iteration)

Reuses hierarchy structure; only nzval arrays change:

1. Rebuild fine-level Dinv via `_amg_build_Dinv_fast!` (single indexed load per
   thread using `diag_ptr`; no row scan, no warp divergence).
2. *(if due per `update_freq`)* For each level: launch the fused Galerkin kernel
   `_amg_galerkin!` (one thread per output nonzero of Ac; reads current A.nzval,
   writes Ac.nzval — no CPU↔device transfers). Then rebuild Dinv for that level.
3. *(if due)* Download coarsest nzval (≤ 2 KB), refill `lu_dense`, re-factorise
   in-place with `lu!(; check=false)`.

### Solve (`solve_system!`)

**PCG path (`krylov = :cg`):**

```
x ← values (warm start)
r ← b − A x
if r0 < atol: return immediately
z ← M⁻¹r  [one V-cycle: L1.b=r, L1.x=0 → L1.x=z; restore L1.r=L1.b after cycle]
p ← z;  rz = rᵀz
for k = 1..itmax:
    Ap ← A p
    α  = rz / (pᵀ Ap)
    x += α p;  r -= α Ap
    check convergence (every step)
    z ← M⁻¹r  [V-cycle; restore L1.r=L1.b after cycle]
    β = (rᵀz_new) / rz;  p = z + β p
```

The `L1.r = L1.b` restore after each V-cycle is essential: `vcycle!` overwrites the
fine-level `L.r` with its internal residual mid-cycle. Without this restore, all
`dot(L1.r, ·)` return ≈ 0 and PCG silently degenerates.

**Richardson path (`krylov = :none`):**

```
x ← values;  r ← b − A x
if r0 < atol: return immediately
for k = 1..itmax:
    x ← vcycle(x, b)    (V-cycle updates x in-place)
    if k==1 or k%5==0: check convergence
```

Both paths exit immediately when the initial residual already satisfies `atol`
(common on warm-started 2nd/3rd PISO correctors).

## Data flow summary

```
         CPU                            Device
──────────────────────────────────────────────────────────
Setup    P_cpu, R_cpu                  P_dev, R_dev (fixed)
         A_cpu (coarsest only)         A_dev per level (nzval changes)
         GalerkinPlan (built here) →→  GalerkinPlan (reused every update!)
         lu_dense, lu_factor           —
──────────────────────────────────────────────────────────
update!  (nothing)                ←    A.nzval (read by Galerkin kernel)
         (nothing)                →→   Ac.nzval (written by Galerkin kernel)
         A_cpu.nzval  ←←← ~2 KB ←←   Ac.nzval (coarsest, for LU only)
──────────────────────────────────────────────────────────
V-cycle  (nothing)                ←→   all work vectors on device
         lu_rhs  ←← b_c (coarsest)    ← device
         lu_rhs  →→ x_c (coarsest)    → device
──────────────────────────────────────────────────────────
```

All work vectors are allocated once at setup; the solve phase performs **zero host
allocations**. Device↔host transfers occur only at the coarsest level (≤ 2 KB per
update, ≤ 400 bytes per V-cycle).

## Exported symbols

| Symbol | Kind | Description |
|---|---|---|
| `AMG` | struct | Solver marker; pass to `SolverSetup` |
| `VCycle` | struct | V-cycle selector |
| `WCycle` | struct | W-cycle selector |
| `JacobiSmoother` | struct | Damped Jacobi smoother |
| `Chebyshev` | struct | Polynomial Chebyshev smoother |
