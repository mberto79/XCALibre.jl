# AMG Solver — Critical Review

Date: 2026-04-17. Based on full read of `src/Solve/AMG/AMG_0`–`AMG_6` and `ext/XCALibre_CUDAExt.jl`.

---

## Executive Summary

- Implementation is architecturally sound: two-tier float precision, synchronous Jacobi, PCG outer loop, lazy Galerkin update, and diag_ptr are all correct.
- W-cycle recursive structure is correct; GPU performance cost is inherent to the algorithm, already warned.
- Three low-risk correctness issues (one dead constant, one conservative spectral estimate, one silent failure mode).
- Two actionable performance wins: batch Galerkin synchronisations, explicit `amg_norm` GPU override.
- cuDSS at coarsest is the only unexplored path that could materially beat the current 0.65× ratio.

---

## File Map

| File | Role |
|------|------|
| `AMG_0_types.jl` | All structs: `AMG`, `AMGWorkspace`, `MultigridLevel`, `LevelExtras`, smoother types |
| `AMG_1_kernels.jl` | KA kernels: SpMV, Jacobi/L1 sweeps, residual, Dinv, RAP |
| `AMG_2_coarsen.jl` | SA and RS coarsening; pairwise fallback |
| `AMG_3_galerkin.jl` | CPU SpGEMM (setup + zero-alloc update), Gershgorin, smooth_P |
| `AMG_4_smoothers.jl` | `amg_smooth!`, Chebyshev, L1-Jacobi; fine-level F32 dispatch |
| `AMG_5_cycle.jl` | V-cycle and W-cycle (fine + coarse); `run_cycle!` dispatch |
| `AMG_6_api.jl` | `amg_setup!`, `update!`, PCG solve, `solve_system!`, diagnostics |

---

## Correctness Issues

### C1 — `_MAX_DENSE_LU_N = 20` is dead (AMG_6_api.jl:10)

The constant is exported and imported by `XCALibre_CUDAExt.jl` (line 65) but is not referenced in any
control-flow path. The actual coarsest-LU guard is `opts.direct_solve_size` (AMG_6_api.jl:334). The
original value `5000` was commented out and replaced with `20` during debugging but never removed. This
creates a false impression that the constant controls something. Action: remove the constant and the
import, or wire it up as the default for `direct_solve_size`.

### C2 — Gershgorin ρ estimate includes diagonal (AMG_3_galerkin.jl:258–265)

`_gershgorin_rho` sums `|a_ij| * |1/a_ii|` for all j including j==i, adding 1 to every row's Gershgorin
radius. Classical Gershgorin for D⁻¹A uses off-diagonal entries only: ρ ≤ 1 + max_i Σ_{j≠i} |a_ij/a_ii|.
The current code gives ρ_est ≥ 2 for any diagonally dominant row. Effect: Chebyshev `λ_max = hi * ρ`
is overestimated → smoothing window is wider than necessary → convergence per sweep is slightly suboptimal
but never diverges. Fix: skip `j == i` in the row sum (one `if j != i` guard).

### C3 — Silent zero-diagonal fallback in Dinv kernels (AMG_1_kernels.jl:120, 137, 174)

All three Dinv kernels use `ifelse(d != 0, 1/d, 1)`. A missing or structurally absent diagonal returns
Dinv[i] = 1 silently, producing incorrect smoothing without any warning. For valid FVM matrices this
never fires, but it hides bugs during development (e.g., wrong `diag_ptr` pointing off-diagonal). Consider
a debug-mode assertion or at least a comment clarifying the intended invariant.

---

## Performance Issues

### P1 — `amg_norm` has no explicit GPU override (AMG_1_kernels.jl:226)

```julia
amg_norm(v) = norm(v)   # "GPU support via extensions"
```

No override exists in `ext/XCALibre_CUDAExt.jl`. For CUDA arrays, `LinearAlgebra.norm` dispatches to
`CUBLAS.nrm2` via CUDA.jl — so CUDA is fine. For AMDGPU or Metal backends, `norm` may fall back to a
CPU reduction, blocking on every PCG convergence check (called twice per iteration). Since the comment
promises an extension override, add one in each GPU ext for correctness, and remove the misleading comment
if CUDA.jl dispatch is sufficient:

```julia
# In XCALibre_CUDAExt.jl:
import XCALibre.Solve: amg_norm
amg_norm(v::GPUARRAY) = CUDA.norm(v)
```

### P2 — Per-level `synchronize()` in `update!` serialises kernel launches (AMG_6_api.jl:402–421)

Each Galerkin+Dinv update pair ends with an explicit `KernelAbstractions.synchronize(backend)`:

```julia
_galerkin_update!(fine, coarse[1], ...)
KernelAbstractions.synchronize(backend)   # ← barrier
_amg_build_smoother_dinv!(...)
KernelAbstractions.synchronize(backend)   # ← barrier
for lvl in 1:n-1
    _galerkin_update!(coarse[lvl], coarse[lvl+1], ...)
    KernelAbstractions.synchronize(backend)   # ← barrier per level
    _amg_build_smoother_dinv!(...)
    KernelAbstractions.synchronize(backend)   # ← barrier per level
end
```

With n_levels coarse levels, this is 2*(n_levels+1) extra barriers per lazy update. Since Galerkin outputs
feed Dinv inputs (data dependency), one sync between them is required. But consecutive level pairs are
independent — the sync after Dinv build for level k is not needed before Galerkin for level k+1. Removing
all but the inter-Galerkin-Dinv barrier per level saves n_levels-1 round-trips. Note: `@elapsed` blocks
on the CPU anyway (it calls `synchronize`), so the timing wrappers already absorb the barriers during
profiling; the benefit is in production (no `@elapsed`).

### P3 — Linear column search in GPU RAP kernels (AMG_1_kernels.jl:316–321, 365–370, 411–415, 454–458)

The four on-device RAP kernels find target columns in the coarse matrix via a linear scan:

```julia
for k in Ac_rowptr[c1]:Ac_rowptr[c1+1]-1
    Ac_colval[k] == c2 && ...
end
```

For unsmoothed P (1 nnz/row), coarse rows have exactly `n_agg` entries and these loops are short. For
smooth_P or after many coarsening levels, coarse rows grow and the O(nnz²) per-warp cost appears. GPU
profiling shows RAP update is cheap at current settings (0.1–0.5 ms), so this is not the bottleneck now.
If smooth_P is ever re-enabled or coarsening is pushed deeper, hash-based column lookup would help.

---

## Design Observations

### D1 — F32 fine smoother is dead code (AMG_4_smoothers.jl:98–111)

The `amg_smooth_fine_f32!` path exists for a GPU-resident F32 fine smoother but this path was closed
after confirming it breaks PCG's SPD requirement (see loop state Iter 3). The CPU fallback (line 99–101)
just calls the F64 path. The dispatch chain still allocates `Dinv_Tc` and `b_Tc` (AMG_6_api.jl:276–279)
when `fine_float == Float32`, which is harmless but wasteful. If F32 fine is permanently closed, remove
the `Dinv_Tc`/`b_Tc` allocation gate and the `_fine_smoother_dispatch!` branch.

### D2 — `sort!(used)` in SpGEMM passes (AMG_3_galerkin.jl:68, 162)

`sparsecsr` requires sorted COO input, so the `sort!(used)` call in each row of `_spgemm` and
`smooth_prolongation` is required by the API. For dense rows (high-connectivity coarse matrices) this
is O(k log k) per row. Alternative for setup-phase SpGEMM: build a sorted integer list directly during
symbolic pass (reuse the `flag` array as a per-row index). Low priority since SpGEMM is setup-only.

---

## Closed Paths (do not revisit)

| Path | Why Closed |
|------|-----------|
| W-cycle on GPU | 2^(n_levels−2) coarse-solve calls per cycle; catastrophic at ≥5 levels |
| F32 fine smoother | Breaks PCG SPD requirement; 1000-iter divergence observed |
| F64 coarse levels | 1.36× slower than F32, no convergence benefit |
| IC0 at coarsest | GPU triangular solve is sequential; slower than 50 Jacobi sweeps |
| CHOLMOD (CPU sparse direct) | 33 ms/call vs ~5 ms Jacobi; PCIe transfer dominates |

---

## Improvement Options

### A — Batch Galerkin synchronisations (P2 fix)
**Impact:** Saves (n_levels-1) barriers per lazy update. For a 5-level hierarchy updated every 2 iters,
saves ~4 barriers every 2 iterations. Measurable on GPU where barriers stall the command queue.
**When it helps:** Multi-level hierarchies, GPU backend, update_freq ≤ 4.
**When it doesn't:** CPU backend (barriers are free), single-level hierarchy.
**Risk:** Low. Data dependencies within each level pair are preserved.

### B — Explicit `amg_norm` override per GPU backend (P1 fix)
**Impact:** Correctness for non-CUDA GPU backends. For CUDA, no change (already uses CUBLAS).
**When it helps:** AMDGPU/Metal backends.
**When it doesn't:** CUDA — already dispatches correctly via CUDA.jl.

### C — Fix Gershgorin diagonal inclusion (C2 fix)
**Impact:** Tighter Chebyshev spectral window. Reduces `λ_max` by ~1.0, allowing tighter `hi` bound
without divergence. Could recover 1–2 iterations on fine levels if Chebyshev is used.
**When it helps:** Chebyshev smoother with `hi` near 1.0 (conservative settings).
**When it doesn't:** JacobiSmoother and L1Jacobi ignore ρ entirely (they use `use_jacobi = true` path).

### D — cuDSS at coarsest level
**Impact:** Replaces 25–50 Jacobi sweeps on the coarsest level with a GPU-resident sparse direct solve.
If the coarsest level is ~1000–5000 rows (typical for RS coarsening with `coarsest_size=50000`), cuDSS
triangular solves would be faster than 50 SpMV sweeps. Loop state identifies this as the only remaining
unexplored path that could push ratio below 0.60.
**When it helps:** Problems where coarse sweeps dominate solve time, coarsest matrix is sparse enough
for cuDSS.
**When it doesn't:** Very small coarsest levels (< 100 rows) where dense LU is already optimal.
**Risk:** Significant implementation effort; requires cuDSS Julia bindings or manual CUDA API calls.

### E — Clean up `_MAX_DENSE_LU_N` dead constant (C1 fix)
**Impact:** Code clarity only. Remove or wire up to `direct_solve_size` default.
**Risk:** None.

---

## Priority Recommendations

1. **(High, low risk)** Fix C1: remove `_MAX_DENSE_LU_N` dead constant and its CUDAExt import.
2. **(High, low risk)** Fix C2: exclude diagonal from Gershgorin row sum (one-line change, benefits Chebyshev).
3. **(Medium, low risk)** Fix P2: batch Galerkin syncs — remove per-level barriers, keep one Galerkin→Dinv barrier.
4. **(Medium, medium risk)** Fix P1: add `amg_norm` GPU override per backend for non-CUDA correctness.
5. **(Low, low risk)** Fix D1: remove dead F32 fine smoother allocations if path is permanently closed.
6. **(High, high effort)** Option D: cuDSS at coarsest — only remaining path to beat 0.60 ratio target.
