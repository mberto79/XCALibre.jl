# AMG Performance Improvement Prompts

## Prompt 1 — Warp-cooperative SpMV (GPU fine-level SpMV throughput)

The current `_amg_spmv!` kernel in `src/Solve/AMG/AMG_1_kernels.jl:6` assigns **one thread
per row**. On GPU this wastes warp lanes on short rows (typical FVM stencil: 5–7 nnz/row in
2D, 20–30 in 3D) and has poor memory coalescing. The fine-level SpMV is the single hottest
call in the AMG hot path — invoked once per smoother sweep, multiple sweeps per V-cycle,
multiple V-cycles per PCG iteration.

Please implement a **warp-cooperative (CSR-Vector) SpMV** replacing `_amg_spmv!` for GPU
backends. Specific requirements:

1. **Kernel design**: Use a workgroup of `WG` threads where each group of `W` consecutive
   threads cooperatively processes one row by partitioning its nonzero range into chunks of
   `W` and performing an intra-group reduction (segmented reduce). `W` should be a tunable
   parameter (try `W = min(32, workgroup)`). The number of rows processed per workgroup is
   `WG ÷ W`.

2. **KernelAbstractions portability**: KernelAbstractions does not expose warp shuffle
   intrinsics directly. Use the `@groupsize`, `@localmem`, `@synchronize`, and
   `@index(Local)` / `@index(Group)` KA macros to implement a shared-memory parallel
   reduction within each row-group. See the KA documentation for `@localmem`. The kernel
   must compile for both CPU (where it degrades gracefully to the one-thread case when
   `WG == W`) and GPU.

3. **Dispatch**: Keep the existing `_amg_spmv!` as the CPU path. Add a new
   `_amg_spmv_warp!` kernel and update `amg_spmv!` in `AMG_1_kernels.jl` to dispatch on
   backend type: `CPU` → existing kernel; `GPU` → new warp kernel. The same GPU dispatch
   should be applied to `_amg_spmv_add!` (the accumulate variant at line 26) and
   `_amg_residual!` (line 157), which are called in the same hot path.

4. **ndrange calculation**: With `W` threads per row and `WG` threads per workgroup, the
   ndrange is `ceil(n_rows, WG÷W) * (WG÷W)` (padded to workgroup multiple). Guard
   out-of-bounds rows with an `if row <= n` check inside the kernel.

5. **Correctness test**: Verify the output of `_amg_spmv_warp!` matches `_amg_spmv!` to
   within `sqrt(eps(Float64))` for a small CSR matrix before running the full solver.

**Reference files:**
- Kernel to replace: `src/Solve/AMG/AMG_1_kernels.jl:6–22` (`_amg_spmv!`)
- Hot-path callers: `AMG_4_smoothers.jl` (Jacobi sweep), `AMG_5_cycle.jl` (vcycle!/wcycle!)
- KA pattern reference: `src/Solve/AMG/AMG_1_kernels.jl` — all existing kernels use
  `_setup(backend, workgroup, ndrange)` which calls `backend, workgroup_size, ndrange`

**Reference paper:** Bell & Garland (2009) "Efficient Sparse Matrix-Vector Multiplication on
CUDA", NVIDIA Technical Report NVR-2008-004. The CSR-Vector format (Section 3.3) is exactly
what to implement.

---

## Prompt 2 — Mixed-precision coarse levels (Float32 below the fine level)

The AMG hierarchy currently uses a single float type `Tv` for all levels (enforced by
`MultigridLevel{Tv,...}` and `Vector{LType}` in `AMGWorkspace`). On GPU, coarse-level work
(SpMV for smoothing + restriction/prolongation) is bandwidth-bound. Converting coarse levels
to **Float32** halves bandwidth for all coarse-level operations and enables tensor-core
acceleration on compatible hardware. The PCG outer loop at the fine level stays in Float64,
so precision loss in coarse corrections is harmless — the PCG iteration corrects it.

Please implement a **two-precision AMG hierarchy**:

### Type system changes (`src/Solve/AMG/AMG_0_types.jl`)

1. Add a second type parameter `Tc` to `AMGWorkspace` (the coarse float type, default
   `Float32`). The fine level stays `Tv = Float64`; all levels from index 2 onward use `Tc`.

2. `MultigridLevel` currently forces all levels into `Vector{LType}` with a single concrete
   type. With two precisions the coarse levels have a different `Tv`, so they cannot share
   the same `LType`. Use one of these two approaches (choose whichever is cleaner):
   - **Two-tier approach** (recommended): store `fine_level::LFType` (a single level, not a
     vector) and `coarse_levels::Vector{LCType}` separately in `AMGWorkspace`. The cycle
     functions receive both and dispatch on level index.
   - **Abstract container** (simpler but adds dispatch): store coarse levels as
     `Vector{AbstractMultigridLevel}` with a narrow abstract interface (`A`, `P`, `R`,
     `Dinv`, `x`, `b`, `r`, `tmp`, `extras`). Less preferred because it reintroduces
     dynamic dispatch in the cycle hot path.

### Hierarchy build (`src/Solve/AMG/AMG_6_api.jl`)

3. In `amg_setup!`, the fine level (index 1) is built with `Tv` (Float64). For level ≥ 2:
   - Cast `A_cpus[i].nzval` to `Vector{Tc}` (Float32) before uploading to device.
   - All work vectors (`x`, `b`, `r`, `tmp`, `Dinv`) are `KernelAbstractions.zeros(backend,
     Tc, n)`.
   - Restriction/prolongation operators `P`, `R` are cast to `Tc` before device upload.
   - The coarsest-level LU dense matrix uses `Tc`.

4. The restriction at each level boundary must cast the residual from `Tv` → `Tc` and the
   prolongation correction from `Tc` → `Tv`. Add two small kernels:
   - `_amg_cast_copy!(dst::AbstractVector{Tc}, src::AbstractVector{Tv})` — element-wise
     type cast, KA kernel, used at the fine→coarse boundary (restrict) and coarse→fine
     boundary (prolongate). This is where precision changes happen; there should be exactly
     two cast kernels per V-cycle.

### Cycle changes (`src/Solve/AMG/AMG_5_cycle.jl`)

5. Modify `vcycle!` (and `wcycle!`) so the fine→coarse transition (levels 1→2) inserts the
   `_amg_cast_copy!` for the restriction RHS and the prolongation correction:
   ```
   # Fine→coarse: cast residual Tv→Tc before restricting
   _amg_cast_copy!(Lc.b_tc, L.r)           # Tv → Tc
   amg_spmv!(Lc.b, Lc_R, Lc.b_tc, ...)    # Tc R * Tc rhs
   # Coarse→fine: cast correction Tc→Tv before prolongating
   _amg_cast_copy!(L.tmp_tv, Lc.x)         # Tc → Tv
   amg_spmv_add!(L.x, L.P, L.tmp_tv, ...) # Tv P * Tv correction
   ```
   All levels from 2 onward communicate entirely in `Tc`.

### Expected benefit

For a 1.7M-DOF fine level with op_complexity 1.8, ~45% of total SpMV work is on coarse
levels. Converting those to Float32 halves bandwidth for that fraction → ~1.3× overall cycle
speedup on bandwidth-bound GPU hardware (A100, H100). On consumer GPUs with lower FP64:FP32
ratio the benefit is larger (up to 1.8×).

**Reference files:**
- Type definitions: `src/Solve/AMG/AMG_0_types.jl` (`MultigridLevel`, `AMGWorkspace`,
  `LevelExtras`)
- Hierarchy build + update: `src/Solve/AMG/AMG_6_api.jl` (`amg_setup!`, `update!`,
  `_galerkin_update!`)
- Cycle functions: `src/Solve/AMG/AMG_5_cycle.jl` (`vcycle!`, `wcycle!`)
- CUDA ext pattern: `ext/XCALibre_CUDAExt.jl` — `_build_sparse_device` shows how to
  construct a GPU sparse matrix from host arrays with type conversion.

**Reference papers:** Anzt et al. (2019) "Adaptive precision in block-Jacobi preconditioning
for iterative sparse linear system solvers", Concurrency and Computation Practice &
Experience; Grützmacher et al. (2020) "A Multiprecision Block-Jacobi Approach for the
Iterative Solution of Linear Systems on GPUs".
