# AMG Performance Improvement Plan — Top 3 Changes for Large-Mesh GPU Runs

## Scope and selection criteria

This document records a critical review of the current AMG solver under
`src/Solve/AMG/` and proposes three performance improvements chosen for their
**probability of producing a measurable wall-clock win on large (≥ 1 M DOF)
GPU runs**, with the expectation that they also do not regress CPU performance
(and where possible, benefit it).

Each proposal below is written as a **self-contained prompt for a separate
implementation session**. They are orthogonal — each attacks a different
bottleneck, so the gains compound multiplicatively. Implement them in the
order listed: #1 unblocks the `update!` path, #2 accelerates every cycle, #3
halves bandwidth at the coarse levels.

---

## Summary of the current AMG hot path

Fine-level flow per PCG iteration (V-cycle + outer CG steps):

1. **V-cycle (pre-smoothing → restriction → recursive coarse solve → prolongation → post-smoothing).**
   Every level fires many small KA kernels (`_amg_jacobi_sweep!`,
   `_amg_residual!`, `_amg_spmv!`, `_amg_spmv_add!`, `_amg_axpby!`,
   `_amg_zero!`, `_amg_copy!`, `_amg_dinv_axpby!`).
2. **Outer PCG step.** Adds another 1× SpMV, 2× dot, 3× AXPY.
3. **`update!` (called once per outer SIMPLE/PISO iteration).**
   Refreshes the Galerkin hierarchy via a CPU round-trip:
   `device → CPU copy → _spgemm_nzval! (Gustavson) → CPU → device upload`,
   for every non-coarsest level.

Dominant costs on a large GPU run, in order:

| Rank | Cost | File/Line | Reason |
|---|---|---|---|
| 1 | CPU Galerkin round-trip | `AMG_6_api.jl:446`–`458` (`_galerkin_update!`) | PCIe transfer + serial CPU SpGEMM; scales with nnz(A) |
| 2 | Fine-level SpMV throughput in Jacobi sweep | `AMG_1_kernels.jl:205` (`_amg_jacobi_sweep!`) | 1 thread per row — poor coalescing for 3D FVM (~25 nnz/row) |
| 3 | Fine-level SpMV in `_amg_spmv!` / `_amg_residual!` / `_amg_spmv_add!` | `AMG_1_kernels.jl:6`, `180`, `26` | Same thread-per-row bottleneck; invoked ≥ 3× per level per cycle |
| 4 | Coarse-level bandwidth | all hot kernels, levels 2…nc | ~45 % of cycle work, all in Float64 |
| 5 | Kernel launch overhead at small coarse levels | everywhere | ~20 launches per level per cycle × ~8 levels |

---

## Ideas considered and rejected (with reasons)

These ideas were considered on first pass but rejected after deeper analysis.
They are documented here so future sessions don't re-propose them.

- **Multicolor Gauss-Seidel (MCGS) smoother.** MCGS converges faster per
  sweep than Jacobi, but (a) it is *not symmetric* unless paired with a
  backward sweep — doubling work — and asymmetry **breaks PCG's
  A-conjugacy**, exactly the constraint that forced the current synchronous
  ping-pong Jacobi (`AMG_4_smoothers.jl:8–9` and the `feedback_amg_julia_type_pitfalls.md`
  memory). Symmetric MCGS (F→B) also pays 2·(ncolors) launches per sweep per
  level, which annihilates the benefit on small coarse levels.
- **cuSPARSE / rocSPARSE SpMV as a drop-in replacement.** Vendor SpMV is
  excellent for `_amg_spmv!`, `_amg_residual!`, and `_amg_spmv_add!`, but it
  **cannot replace `_amg_jacobi_sweep!`** — that kernel fuses a row traversal
  with a skip-diagonal test and an AXPBY, and cuSPARSE has no matching
  primitive. Using cuSPARSE for only 3 of the 4 hot kernels forces you to
  maintain two code paths; implementing a warp-cooperative KA kernel (item 2
  below) with reusable inner-loop logic covers *all four* kernels at once and
  is portable across CUDA / ROCm / Metal. Vendor SpGEMM, however, is still
  the right answer for the Galerkin product (item 1 below) — the reasoning
  is different and narrower there.
- **Pipelined (Ghysels–Vanroose) PCG.** Hides dot-product latency behind
  SpMV, but on a single GPU the gain is negligible because there is no MPI
  all-reduce — the bottleneck is local memory bandwidth, not reduction latency.
- **Fused GPU Galerkin with explicit index plan.** Already tried and removed
  in commit 05492113 due to memory explosion (`O(nnz_A · stencil²)` entries).
  Item 1 below takes a *different* GPU approach that avoids this pitfall.
- **Parallel coarsening on the GPU.** Setup runs once per hierarchy build,
  which is a tiny fraction of total runtime for production runs with many
  outer iterations. Not a priority.

### Free-standing zero-effort tuning experiment (not a prompt — just try it)

The `Chebyshev(; degree=2)` smoother is already implemented
(`AMG_4_smoothers.jl:44`) but is **not the default** — the default is
`JacobiSmoother(2, 2/3)`. Chebyshev(2) applies exactly two SpMVs per call
(identical work to two Jacobi sweeps) but is spectrally optimal on the
high-frequency eigenspace, typically reducing total V-cycle count by
10–30 %. Switching the default is a one-line change in `AMG_0_types.jl:171`
and should be measured before and after the three larger changes below.

---

# Prompt 1 — GPU-side Galerkin update (eliminate the CPU round-trip)

## Goal and why this is #1

The current `update!` path at `src/Solve/AMG/AMG_6_api.jl:269` calls
`_galerkin_update!` for every non-coarsest level
(`src/Solve/AMG/AMG_6_api.jl:446–458`). That function:

1. **Downloads** the fine-level `A.nzval` from device to the CPU mirror
   (`copyto!(ex.A_cpu.nzval, nzval_dev)`).
2. Runs **two CPU SpGEMMs** in-place via `_spgemm_nzval!`
   (`src/Solve/AMG/AMG_3_galerkin.jl:213`) to compute `T = A·P` then
   `Ac = R·T`.
3. **Uploads** the result back to the device (`Lc.A.nzval`).

For a 1.7 M-DOF fine level with ~20 nnz/row, this is on the order of
30–50 ms per update — much more than a whole V-cycle. With
`update_freq = 2` (the current default) it runs every other outer iteration
and dominates total solver time on any modern GPU. It is also the reason
`update_freq = 1` is not usable in practice even though it gives the best
AMG convergence.

The earlier attempt at a GPU-resident RAP kernel (commit 05492113) built an
explicit index plan of `(nzi_R, nzi_A, nzi_P, nzi_C)` quadruples and
exploded in memory because the plan scales as `O(nnz_A · stencil²)`.
This prompt avoids that trap.

## What to implement

A **GPU-resident numeric RAP update** that writes directly into
`Lc.A.nzval` without going near the CPU. Two alternative implementations
are acceptable; start with (A), fall back to (B) if cuSPARSE version is
too old.

### Strategy (A): cuSPARSE / rocSPARSE SpGEMM *reuse* API (preferred)

Both `cusparseSpGEMMreuse_*` (CUDA ≥ 11.4 / cuSPARSE ≥ 11.7) and
`rocsparse_spgemm_reuse` (ROCm ≥ 5.2) support a three-phase pattern:

1. **Work estimation** — once, at `amg_setup!` time, for each non-coarsest
   level, run `cusparseSpGEMMreuse_workEstimation` + `_nnz` on the pair
   `(A, P)` and again on `(R, AP)` to obtain the symbolic pattern and
   external buffer sizes.
2. **Compute_copy** — once, at setup time, allocate the persistent
   `externalBuffer2` and populate the Ac structure (`rowptr`, `colval`).
3. **Numeric update (reuse)** — every `update!` call, invoke
   `cusparseSpGEMMreuse_compute` with the existing buffers; this re-runs
   only the numeric multiply and writes new values into `Ac.nzval`.

This is exactly the pattern Ginkgo and hypre use on GPU. **The symbolic
phase runs once per hierarchy build, not per update**, so the per-update
cost drops to the vendor's tuned numeric kernel — typically 5–10× faster
than the CPU Gustavson implementation plus zero PCIe traffic.

### Strategy (B): streaming row-by-row RAP kernel (fallback if cuSPARSE version is too old or ROCm path lags)

Write a KA `@kernel` whose work item is **one row of Ac**. For row `i`:

```
for each nz r in row i of R:        # iterate aggregate members
    k = colval_R[r];  rvalue = nzval_R[r]
    for each nz a in row k of A:
        j_mid = colval_A[a];  avalue = nzval_A[a]
        for each nz p in row j_mid of P:
            j_out = colval_P[p];  pvalue = nzval_P[p]
            acc[j_out] += rvalue * avalue * pvalue   # scatter into Ac[i,:]
```

The scatter target must match the *pre-computed* `Ac.colval` for row `i`.
Because unsmoothed P has exactly one nnz per row, this reduces to a small
bounded loop (typical Ac row has ≤ 30 entries in 3D FVM). Do the scatter
via a small `@localmem` dense bin of size
`max_nnz_per_row(Ac)` per work item, indexed by a tiny lookup map
`col_to_slot[Ac_col] → local slot` — analogous to the compact CPU
accumulator already in `_spgemm_nzval!`. The key point that avoids the
previous memory-explosion failure: **no explicit triple-product index
plan is stored**; each thread recomputes its contributions on the fly and
uses only its row's local accumulator.

Memory footprint: `O(n_coarse · max_row_nnz · sizeof(Tv))` in shared / local
memory per workgroup, which is bounded (< 4 KB per row) and fits in any GPU.

## Detailed requirements

1. **Files to edit:**
   - `src/Solve/AMG/AMG_6_api.jl` — rewrite `_galerkin_update!` to dispatch
     on backend type. CPU path unchanged; GPU path calls the new kernel.
   - `src/Solve/AMG/AMG_0_types.jl` — add two optional fields to
     `LevelExtras`: `spgemm_buffer1::Union{Nothing, AbstractVector{UInt8}}`
     and `spgemm_buffer2::Union{Nothing, AbstractVector{UInt8}}` to hold
     the persistent cuSPARSE workspace.
   - `ext/XCALibre_CUDAExt.jl` — add a `_amg_galerkin_update_gpu!(Lc, L,
     backend)` method that wraps `cusparseSpGEMMreuse_compute`. Guard the
     include with `@static if CUDA.CUSPARSE.version() >= v"11.7"`.
   - `ext/XCALibre_AMDGPUExt.jl` (or equivalent) — analogous wrapper using
     `rocsparse_spgemm_reuse`. If ROCm version is too old, fall back to
     Strategy (B).
   - `src/Solve/AMG/AMG_1_kernels.jl` — if Strategy (B) is used, add the
     new streaming kernel `_amg_galerkin_rap_row!` alongside existing
     kernels. Follow the `_setup(backend, workgroup, ndrange)` idiom.

2. **Setup phase changes (`amg_setup!` in `AMG_6_api.jl:48`):**
   For each non-coarsest level `i`, after `P_cpu`, `R_cpu`, `AP_cpu`, and
   `Ac_cpu` are built on CPU, additionally:
   - Allocate device-resident `A_dev`, `P_dev`, `R_dev`, `AP_dev`, `Ac_dev`
     matching the CPU structures (AP_dev and Ac_dev get their symbolic
     pattern from the CPU SpGEMM output, numeric values from the first
     upload).
   - If Strategy (A): run the cuSPARSE work estimation on `(A_dev, P_dev)`
     and `(R_dev, AP_dev)` and store the resulting `externalBuffer2` in
     `extras.spgemm_buffer1/2`.
   - If Strategy (B): upload `Ac.rowptr`/`Ac.colval` and `AP.rowptr`/
     `AP.colval` to the device as `Int32` arrays; build a per-row
     `col_to_slot` lookup map on CPU and upload it.

3. **Update phase (`update!` in `AMG_6_api.jl:269`):**
   - Replace the current `_galerkin_update!(L, Lc, backend)` call with
     a backend-dispatched version. On GPU, no data leaves the device.
     On CPU, keep the current path.
   - The fine-level `Dinv` rebuild at line 281 and the coarsest-level LU
     path at lines 295–302 stay unchanged — the LU is on ~50 rows so the
     PCIe cost is negligible.

4. **Correctness test:**
   - In a small unit test, build a hierarchy from a 2D Laplacian
     (`n = 64²`), run the new GPU update, download `Lc.A.nzval`, compare
     element-wise against the existing CPU path output. Tolerance
     `sqrt(eps(Float64))`.
   - Run the existing AMG solver test suite (`test/unit_test_*.jl`) and
     confirm PCG iteration counts match to within ±1 of the CPU baseline.

5. **Benchmark target:** on a 1–2 M-DOF 3D FVM pressure solve, measure
   `update!` wall-time before and after. Expected: 5–10× reduction.
   Whole-solver speedup for `update_freq = 2` in SIMPLE: 15–30 %.

## Why this might not work (honest failure modes)

- **cuSPARSE SpGEMM reuse still allocates nontrivial `externalBuffer2`** —
  typically 2–3× `nnz(Ac)` bytes. For a 1.7 M-DOF fine with nnz ≈ 13 M,
  Ac has nnz ≈ 9 M, buffer ≈ 150 MB. Acceptable on modern GPUs but must
  be tracked and freed on `amg_setup!` re-entry.
- **rocSPARSE lags cuSPARSE** — the reuse API may be missing on older
  ROCm. If so, use Strategy (B) unconditionally on AMD.
- **The Galerkin update is already skipped when `update_freq > 1`.** If a
  user runs with `update_freq = 5`, the per-outer-iteration amortised cost
  is already low and this change buys less than advertised. The headline
  win is for users who *wanted* `update_freq = 1` but couldn't afford it.
- **Strategy (B)'s local-memory usage scales with `max_nnz_per_row(Ac)`.**
  On pathological meshes with a few "hub" aggregates (e.g. very
  non-uniform meshes) that single maximum can be large and force
  workgroup size down. Detect this at setup and fall back to the CPU path
  for affected levels.

## Reference material

- [cuSPARSE SpGEMM reuse API](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spgemm-reuse)
- Ginkgo's GPU AMG: Anzt et al. (2020), "Ginkgo: A high performance
  numerical linear algebra library", JOSS 5(52).
- hypre's `hypre_ParCSRMatrixRAPKT` on device.
- NVIDIA AMGX source — `amgx/src/cuda/gpu_amg.cu` for a production RAP
  implementation.

---

# Prompt 2 — Warp-cooperative CSR-Vector SpMV for all hot-path KA kernels

## Goal and why this is #2

Four kernels in `src/Solve/AMG/AMG_1_kernels.jl` iterate the CSR row loop
`for nzi in rowptr[row]:(rowptr[row+1]-1)` with **one thread per row**:

- `_amg_spmv!`            (line 6)   — restriction/prolongation SpMV
- `_amg_spmv_add!`        (line 26)  — prolongation-accumulate
- `_amg_residual!`        (line 180) — residual
- `_amg_jacobi_sweep!`    (line 205) — *the* hottest kernel (fused SpMV + AXPBY)

On GPU, a warp of 32 threads is issued per row, but each thread processes
only one row. For a 3D FVM matrix with ~25 nnz/row the inner loop retires
in ~25 cycles while the warp still has 32 lanes — poor instruction-level
parallelism, poor memory coalescing (adjacent threads load
non-contiguous `colval[nzi]` + `x[colval[nzi]]`), and the hottest kernel
of the whole AMG pipeline runs at perhaps 20–30 % of peak HBM bandwidth.

The classical fix (Bell & Garland 2009, used by AMGX, hypre, Ginkgo) is
**CSR-Vector**: assign a small team of `W` threads to each row, have them
stride through the nnz of that row in chunks of `W`, and reduce the
partial sums via shared memory. With `W` tuned to the average row nnz,
this recovers near-peak bandwidth.

**Why this targets all four kernels at once:** all four contain the same
inner row-loop template. Implementing one shared inline helper (or one
parametric `@kernel` with a small lambda for the row postlude) lets you
upgrade *all four* kernels — not just standalone SpMV — and critically
includes the Jacobi-sweep kernel that cuSPARSE cannot cover.

## What to implement

### Core design

A new set of KA kernels `_amg_spmv_warp!`, `_amg_spmv_add_warp!`,
`_amg_residual_warp!`, `_amg_jacobi_sweep_warp!`, each using a
**warp-cooperative row reduction** built on KA primitives:

- Workgroup size `WG` (tune: 128 or 256).
- Each row is processed by a group of `W` threads (`W ∈ {4, 8, 16, 32}`,
  chosen per level at setup time from average nnz/row).
- Each workgroup processes `WG ÷ W` rows.
- The `W` threads of a row-group partition the row's nnz range into
  strided chunks of `W`, accumulate a partial sum in a register, then
  perform a shared-memory reduction via `@localmem` + `@synchronize` to
  produce the row sum.

KA does **not** expose warp shuffle intrinsics directly. The reduction
must be implemented with shared memory:

```julia
@kernel function _amg_spmv_warp!(y, rowptr, colval, nzval, x, ::Val{W}, ::Val{WG}) where {W,WG}
    tid        = @index(Local)          # 1..WG
    gid        = @index(Group)          # 1..ngroups
    lane       = mod1(tid, W)           # 1..W inside the row-group
    row_local  = (tid - 1) ÷ W + 1      # 1..(WG÷W)
    row        = (gid - 1) * (WG ÷ W) + row_local
    n_rows     = length(y)

    # Shared memory for the reduction: (WG÷W) × W slots
    shared     = @localmem eltype(nzval) (WG,)

    acc = zero(eltype(nzval))
    if row <= n_rows
        rs = rowptr[row]; re = rowptr[row+1] - 1
        k  = rs + lane - 1
        while k <= re
            acc += nzval[k] * x[colval[k]]
            k += W
        end
    end

    shared[tid] = acc
    @synchronize

    # Tree reduction inside the W-wide team
    s = W ÷ 2
    while s > 0
        if lane <= s
            shared[tid] += shared[tid + s]
        end
        @synchronize
        s ÷= 2
    end

    if lane == 1 && row <= n_rows
        y[row] = shared[tid]   # or b[row] - shared[tid] for residual
    end
end
```

(Exact structure and the Val-based `W`/`WG` dispatch pattern match the
existing patterns described in `memory/reference_kernelabstractions.md`.)

The `_amg_jacobi_sweep_warp!` variant has the same reduction skeleton but
skips `j == row` inside the inner loop and, after the reduction, the
lane-1 thread emits
`x_new[row] = omega * Dinv[row] * (b[row] - acc) + (1 - omega) * x[row]`.

### Kernel dispatch

Add a backend-dispatched wrapper `amg_spmv!(y, A, x, backend, workgroup)`
at `src/Solve/AMG/AMG_1_kernels.jl:17`:

- `backend isa CPU`  → existing thread-per-row kernel.
- `backend isa GPU`  → new warp kernel with `W` chosen from the level's
  average nnz/row at setup time and stored in a new field
  `extras.spmv_W::Int32`.

The same dispatch is applied to `amg_spmv_add!`, `amg_residual!`, and
`amg_jacobi_sweep!`.

### `W` selection (per level, at setup)

At `amg_setup!` time, after each level's `A_cur` is available, compute
`avg_nnz = nnz(A_cur) / size(A_cur, 1)` and pick:

| avg nnz / row | W |
|---|---|
| ≤ 4    | 2  |
| 5–8    | 4  |
| 9–16   | 8  |
| 17–32  | 16 |
| > 32   | 32 |

Store on `LevelExtras`. This is the "CSR-Adaptive" strategy from Greathouse
& Daga (AMD, SC14) — per-row would be better but per-level is much simpler
and captures most of the benefit.

### ndrange

With `W` threads per row and `WG` threads per workgroup, `WG` must be a
multiple of `W`, and the ndrange is
`cld(n_rows, WG ÷ W) * WG` (padded to workgroup multiple). Out-of-bounds
rows are guarded by `if row <= n_rows` as shown above.

### CPU correctness

The warp kernel compiles on CPU — just make `W = 1` on CPU so the path
degenerates to one thread per row (identical to the current kernel).
This means both backends can share the warp-kernel code; no need to
maintain two parallel copies. The CPU path keeps the existing
`_amg_spmv!` for clarity and as a well-validated reference.

## Detailed requirements

1. **Files to edit:**
   - `src/Solve/AMG/AMG_1_kernels.jl` — add the four new warp kernels
     alongside the existing ones; refactor the inner row-loop into a
     small helper `@inline _warp_row_reduce(...)` so all four kernels
     share it. Update the `amg_*!` wrappers to dispatch on backend.
   - `src/Solve/AMG/AMG_0_types.jl` — add `spmv_W::Int32` to
     `LevelExtras`. Default `Int32(1)` (CPU path).
   - `src/Solve/AMG/AMG_6_api.jl` — in `amg_setup!`, compute `spmv_W` from
     `avg_nnz` for each level and store it in `extras`.

2. **Workgroup hardware tuning:** expose `WG` as a module constant with
   a good default (128 on NVIDIA, 256 on AMD). Do not make it a user
   option.

3. **Correctness test (MANDATORY before benchmarking):**
   Build a small CSR matrix from a 2D Poisson 5-stencil on `n = 64²`,
   run both `_amg_spmv!` and `_amg_spmv_warp!` with the same input, and
   assert `norm(y_warp - y_reference) < sqrt(eps(Float64)) * norm(y_reference)`.
   Repeat for `_amg_jacobi_sweep!` and `_amg_residual!`. Do the same on a
   3D example with ~25 nnz/row.

4. **Benchmark target:** on the fine level of a 1–2 M-DOF 3D FVM pressure
   solve, measure a single V-cycle (`@belapsed` excluding the PCG outer
   loop) before and after. Expected per-cycle speedup: 1.4–1.8× on
   consumer GPUs, 1.3–1.5× on A100/H100 (already higher baseline
   bandwidth utilisation on those chips).

## Why this might not work (honest failure modes)

- **Shared-memory bank conflicts.** The tree reduction as written touches
  `shared[tid + s]` where `s` is a power of two — classic bank conflicts
  for `W ≥ 32`. Mitigate by padding the shared array stride to `WG + 1`
  or by using a sequential (rather than tree) reduction for small `W`.
- **`@synchronize` body-splitting rule.** KA's `@synchronize` requires
  structured control flow (see `memory/reference_kernelabstractions.md`).
  Loops with data-dependent trip counts (`while k <= re`) inside a
  synchronised region are allowed, but the reduction loop must be
  unrolled or written with an unconditional iteration count. Use a
  `for s in (W÷2, W÷4, ..., 1)` loop not a `while`.
- **W=2 and W=4 may be slower than thread-per-row on short rows.** The
  sync overhead dominates. At setup, also apply a fallback: if
  `avg_nnz ≤ 4`, use the scalar CPU-style kernel on GPU as well. (This
  is common on 2D fine-level meshes.)
- **Coarse levels have small row counts.** A level of 200 rows launches
  only a few workgroups and under-occupies the GPU regardless of the
  kernel design. For such levels the benefit is zero; they run at launch
  overhead speed either way. That's fine — the fine level dominates and
  is where the win comes from.
- **The kernel must also correctly handle zero-length rows.** Guard
  `if rs > re` at kernel entry (can happen at the coarsest level before
  the hierarchy stabilises).

## Reference material

- Bell & Garland (2009), "Implementing Sparse Matrix-Vector Multiplication
  on Throughput-Oriented Processors", SC09. The CSR-Vector kernel in
  Section 3.3 is the canonical source.
- Greathouse & Daga (2014), "Efficient Sparse Matrix-Vector Multiplication
  on GPUs Using the CSR Storage Format", SC14. Per-row adaptive W.
- AMGX open source: `amgx/base/src/cuda/sparse_ops.cu` — the
  `csr_spmv_v2_kernel_strided` template is the production reference.
- `memory/reference_kernelabstractions.md` — internal notes on KA
  `@localmem`, `@synchronize`, and `Val{W}` dispatch.

---

# Prompt 3 — Mixed-precision AMG hierarchy (Float32 coarse levels)

## Goal and why this is #3

The hierarchy currently uses a single float type `Tv = Float64` across all
levels (enforced by `MultigridLevel{Tv,…}` at `AMG_0_types.jl:255` and the
`Vector{LType}` constraint at `AMG_0_types.jl:291`). On GPU every coarse
level is **bandwidth-bound**: a coarse-level SpMV saturates HBM long
before it saturates FMA throughput.

Converting all levels from index 2 onward to **Float32** halves the
bandwidth needed for every coarse-level `nzval` and work vector (`x`, `b`,
`r`, `tmp`, `Dinv`), and doubles SIMD/tensor-core throughput on
compatible hardware. The PCG outer loop at the fine level stays in
Float64, so the V-cycle is still used as a Float64-consistent
preconditioner — precision loss inside the coarse correction is
corrected by the outer CG iteration and does not affect the final
solution accuracy (this is the same pattern used by AMGX's
`MIXED_PRECISION` mode and by Ginkgo's `amgx_mgsetup_mixed`).

For a 1.7 M-DOF fine level with op_complexity ≈ 1.8, **~45 % of total
SpMV work is at the coarse levels**. Halving bandwidth for that fraction
gives a ~1.3× cycle-time speedup on A100/H100, and up to ~1.6× on
consumer GPUs where the FP64:FP32 ratio is worse (1:16 on GeForce). On
CPU the benefit is smaller but nonzero — AVX-512 doubles effective
throughput in Float32 for bandwidth-bound loops.

## What to implement

### Type-system change (`src/Solve/AMG/AMG_0_types.jl`)

1. Add a **second type parameter `Tc`** (coarse float type) to
   `AMGWorkspace`, defaulting to `Float32`:

   ```julia
   mutable struct AMGWorkspace{LFType, LCType, Vec, Opts<:AMG}
       fine_level   :: LFType
       coarse_levels:: Vector{LCType}
       x            :: Vec
       opts         :: Opts
       setup_valid  :: Bool
       setup_count  :: Int
       update_count :: Int
       x_pcg        :: Vec
       p_cg         :: Vec
   end
   ```

   `LFType` is a `MultigridLevel{Tv,…}` (Float64), `LCType` is a
   `MultigridLevel{Tc,…}` (Float32). Both use the same parametric struct
   — the split is only in the workspace.

   This is the **two-tier approach**: single fine level (not wrapped in
   a vector) plus a homogeneous vector of coarse levels. It keeps the
   cycle hot path dispatch-free: each function signature is
   `(fine, coarse_levels, …)` and all accesses are concrete.

2. `MultigridLevel` itself is unchanged. It already takes `Tv` as a type
   parameter; both `Tv = Float64` (fine) and `Tv = Float32` (coarse)
   are supported without further modification.

### Hierarchy build (`src/Solve/AMG/AMG_6_api.jl`)

3. In `amg_setup!`, the fine level (index 1) is built exactly as today
   with `Tv = Float64`. For each level `i ≥ 2`:
   - Cast `A_cpus[i].nzval` from `Vector{Float64}` to `Vector{Tc}`
     (Float32) before uploading to the device.
   - Allocate all work vectors (`x`, `b`, `r`, `tmp`, `Dinv`) via
     `KernelAbstractions.zeros(backend, Tc, n)`.
   - Cast `P_cpus[i].nzval` and `R_cpus[i].nzval` to `Tc` before device
     upload. The `colval` and `rowptr` stay `Int32`.
   - The coarsest-level LU dense matrix (`extras.lu_dense`) uses `Tc`.
   - Use `_build_sparse_device(backend, rowptr, colval, nzval_Tc, m, n)`
     exactly as in the existing `_csr_to_device` path — that helper
     already handles arbitrary `Tv`.

4. In `update!`, the fine-level `Dinv` rebuild is unchanged (Float64).
   For coarse levels, the `_amg_build_smoother_dinv!` call at
   `AMG_6_api.jl:291` receives a `Tc`-typed `Lc.Dinv` and `Lc.A`, and
   the existing kernel already dispatches on `eltype(nzval)`.
   **Verify** this compiles and produces correct results — it should
   (KA kernels are generic over element type) but test first.

### Cast kernels (`src/Solve/AMG/AMG_1_kernels.jl`)

5. Add one new KA kernel and its wrapper:

   ```julia
   @kernel function _amg_cast_copy!(dst, src)
       i = @index(Global)
       @inbounds dst[i] = convert(eltype(dst), src[i])
   end

   function amg_cast_copy!(dst, src, backend, workgroup)
       n = length(dst)
       kernel! = _amg_cast_copy!(_setup(backend, workgroup, n)...)
       kernel!(dst, src)
   end
   ```

   This kernel handles both directions: `Float64 → Float32` at the
   restriction boundary and `Float32 → Float64` at the prolongation
   boundary.

### Cycle modification (`src/Solve/AMG/AMG_5_cycle.jl`)

6. Modify `vcycle!` and `wcycle!` so the fine→coarse and coarse→fine
   transitions cast once at the boundary between level 1 (Float64) and
   level 2 (Float32):

   ```julia
   # Fine→coarse (level 1 → level 2):
   #   r_fine in Float64, need b_coarse in Float32
   amg_cast_copy!(L1.r_Tc, L1.r, backend, workgroup)   # Float64 → Float32
   amg_spmv!(L2.b, L2.R, L1.r_Tc, backend, workgroup)  # all Float32
   amg_zero!(L2.x, backend, workgroup)

   # ... recursive coarse solve, all Float32 ...

   # Coarse→fine (level 2 → level 1):
   #   x_coarse in Float32, prolongate into level 1.x (Float64)
   amg_spmv!(L2.tmp_Tc, L1.P, L2.x, backend, workgroup)  # P is Tc, x is Tc, output Tc
   amg_cast_copy!(L1.tmp, L2.tmp_Tc, backend, workgroup) # Float32 → Float64
   amg_axpy!(L1.x, L1.tmp, one(Float64), backend, workgroup)
   ```

   Note that `L1.P` must be in Float32 so that the `P * x_c` product
   keeps everything on the Float32 bandwidth curve. `L1.P` lives in the
   *fine* level's struct (per current convention where `level.P` is the
   prolongation *from* the next coarser level). Decide whether `L1.P`
   should be Float64 or Float32 — the consistent choice is **Float32**
   because P is only used at the level-1→level-2 boundary and Float32 is
   correct for the coarse input and output of that SpMV. Document this
   clearly.

7. Add two scratch vectors to `LFType`'s `LevelExtras`:
   `r_Tc::Union{Nothing, Vec_Tc}` and `tmp_Tc::Union{Nothing, Vec_Tc}` —
   the two Float32 boundary buffers used in the cast above. Allocate at
   setup time. These are the **only** two extra allocations for the
   mixed-precision path.

### PCG outer loop (`src/Solve/AMG/AMG_6_api.jl:309`)

8. The PCG loop stays entirely in Float64 (fine-level operators only).
   The V-cycle is invoked once per PCG step; the cast kernels are
   called inside the V-cycle so PCG is oblivious to the mixed precision.
   **No changes needed in `_amg_pcg_solve!`**. Confirm by inspection.

### Host-side CPU SpGEMM update path

9. The CPU Galerkin update (`_galerkin_update!` with
   `_spgemm_nzval!`) must also run in Float32 for levels ≥ 2. That means
   `extras.A_cpu`, `extras.AP_cpu`, `extras.Ac_cpu`, and
   `extras.cpu_tmps` must be stored in `SparseMatrixCSR{1, Tc, Int}` for
   those levels. The existing template `_spgemm_nzval!` is generic on
   `Tv` — no code change, only the concrete types flowing through.

   **Interaction with Prompt 1:** if Prompt 1 lands first (GPU Galerkin),
   then this CPU-side change becomes moot — the GPU path is precision-
   parametric by construction. Implement Prompt 3 to handle both cases:
   check `backend` in `_galerkin_update!` and use the Tc path on both
   branches.

## Detailed requirements

1. **Scope of type change:** fine level is **always** Float64 (required
   for outer PCG). Coarse levels are **always** Float32 under this
   change (not user-configurable — a second type parameter adds
   complexity without clear benefit).

2. **Correctness test:**
   - Build a hierarchy from a 2D Laplacian `n = 256²`, solve to
     `rtol = 1e-8` with the mixed-precision path, compare iteration
     count and final residual against the pure-Float64 path. Expected:
     same iteration count ±1, final residual within a factor of 2.
   - Build a hierarchy from the 3D cylinder pressure matrix used in
     the existing benchmarks, same comparison.
   - **Guard against catastrophic precision loss**: at setup, compute
     the fine-level diagonal magnitude max and assert that Float32
     dynamic range is sufficient (i.e. max/min < 10⁶). If not, fall
     back to an all-Float64 hierarchy and log a warning.

3. **Benchmark target:** on a 1–2 M-DOF 3D FVM pressure solve on an
   A100, measure V-cycle wall time before and after. Expected:
   1.25–1.45× per-cycle speedup, ~50 % reduction in GPU memory
   footprint for the coarse hierarchy.

## Why this might not work (honest failure modes)

- **Mixed-precision AMG can hurt PCG convergence** if the Float32
  coarse-level correction is sufficiently inaccurate that PCG restarts
  or stalls. In practice this has been measured by AMGX and hypre to be
  *not* a problem for SPD pressure Laplacians (the fine-level Float64
  residual update recovers any precision loss). Monitor the PCG
  iteration count in the test suite — if it grows > 10 %, investigate.
- **Non-uniform meshes with large dynamic range in diagonal magnitudes**
  (ratio > 10⁶) overflow/underflow in Float32. Catch at setup via the
  guard above; fall back to all-Float64.
- **Type surgery in the cycle hot path risks accidental boxing.** The
  `AMGWorkspace{LFType, LCType, …}` split must be *fully concrete* —
  every `@code_warntype` of `vcycle!` must show `Body::Nothing` with no
  red. Validate with a small test before benchmarking.
- **CPU speedup is modest.** Julia's SIMD loop vectoriser handles
  Float32 well only when loops are simple; the current CSR inner loop
  with indirect `colval` access is gather-limited, not arithmetic-
  limited. Expect ~10–20 % CPU speedup, not 2×.
- **Adds a new axis of testing.** All existing tests run Float64; a new
  set of `_mixed_precision` tests is needed to exercise the casts.

## Reference material

- Anzt et al. (2019), "Adaptive precision in block-Jacobi preconditioning
  for iterative sparse linear system solvers", Concurrency and
  Computation Practice & Experience.
- Grützmacher et al. (2020), "A Multiprecision Block-Jacobi Approach for
  the Iterative Solution of Linear Systems on GPUs", IJHPCA.
- NVIDIA AMGX `config/AMG_MIXED_PRECISION.json` — the production
  mixed-precision hierarchy config.
- `ext/XCALibre_CUDAExt.jl` — `_build_sparse_device` already supports
  arbitrary `nzval` element types; no change required there.

---

## Summary table

| # | Change | Bottleneck hit | GPU speedup (expected) | CPU speedup | Risk | Depends on |
|---|---|---|---|---|---|---|
| 1 | GPU Galerkin via cuSPARSE SpGEMM reuse (or streaming RAP kernel) | CPU Galerkin round-trip (cost #1) | 5–10× on `update!`, 15–30 % on whole solver | None (CPU path unchanged) | Medium (memory, ROCm version gap) | — |
| 2 | Warp-cooperative CSR-Vector SpMV for all 4 hot kernels | Fine + coarse SpMV throughput (costs #2, #3) | 1.4–1.8× per V-cycle | Neutral (W=1 falls back) | Medium (bank conflicts, sync) | — |
| 3 | Float32 coarse levels | Coarse-level bandwidth (cost #4) | 1.25–1.45× per V-cycle | 1.1–1.2× | Medium (type surgery, precision guards) | — |

The three changes are **orthogonal** — they target independent
bottlenecks and can be implemented in any order, though #1 is the
highest-impact single change for large-mesh GPU workloads and should
land first. All three together should roughly **halve total solver wall
time on a 1–2 M-DOF 3D FVM pressure solve**, with the Galerkin-update
change alone responsible for most of the gain at typical `update_freq`
values.

**Before implementing any of these**, run the one-line experiment of
swapping the default smoother from `JacobiSmoother` to `Chebyshev(; degree=2)`
(`src/Solve/AMG/AMG_0_types.jl:171`) and measure the iteration-count
change on a representative solve. If Chebyshev reduces PCG iterations
by ≥ 15 %, make it the new default — it is a free win that stacks on top
of all three prompts above.
