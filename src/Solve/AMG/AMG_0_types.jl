export AMG, VCycle, WCycle, Chebyshev, L1Jacobi

# ─── Cycle markers ────────────────────────────────────────────────────────────

abstract type AMGCycle end
struct VCycle <: AMGCycle end
struct WCycle <: AMGCycle end

# ─── Chebyshev polynomial smoother ────────────────────────────────────────────

"""
    Chebyshev(; degree=2, lo=0.3, hi=1.1)

Polynomial Chebyshev smoother for use inside an AMG hierarchy.

# Fields
- `degree` — polynomial degree (number of SpMV applications per sweep).
- `lo`, `hi` — fraction of the estimated spectral radius defining the dampening window.
"""
struct Chebyshev{F<:AbstractFloat} <: AbstractSmoother
    degree::Int
    lo::F
    hi::F
end
Adapt.@adapt_structure Chebyshev

Chebyshev(; degree::Int=2, lo=0.3, hi=1.1) = begin
    F = promote_type(typeof(lo), typeof(hi))
    Chebyshev{F}(degree, F(lo), F(hi))
end

# ─── l1-Jacobi smoother ───────────────────────────────────────────────────────

"""
    L1Jacobi(; omega=1.0)

l1-Jacobi smoother for use inside an AMG hierarchy.

Scales the Jacobi update by `1/||a_i||_1` (the l1 row norm, summing all
absolute values in the row) instead of `1/a_ii`. For FVM M-matrices this
bounds the spectral radius of `D_l1⁻¹A` to ≤ 1 by construction.

# Fields
- `omega` — relaxation factor. Default `1.0`.

# When to expect improvement

The benefit of l1-Jacobi is primarily in **distributed-memory** (MPI) AMG, where each
process sees only its local sub-matrix. Missing off-processor entries make the local
diagonal `a_ii` smaller than the true global row sum, causing standard Jacobi to diverge
or converge slowly. The l1 norm implicitly accounts for those missing entries.

**In serial / shared-memory runs on near-isotropic FVM M-matrices** (e.g. pressure
Laplacian), the row satisfies `a_ii ≈ Σ_{j≠i}|a_ij|`, so `||a_i||_1 ≈ 2·a_ii`. With
`omega=1.0` this halves the effective relaxation and converges *slower* than
`JacobiSmoother(omega=2/3)`. Use `omega=4/3` to recover equivalent performance, or
simply prefer `JacobiSmoother` for serial runs on isotropic operators.

# Reference
Baker, Falgout, Kolev, Yang — "Multigrid Smoothers for Ultraparallel Computing",
SIAM J. Sci. Comput. 2011. This is the default smoother in hypre's BoomerAMG.
"""
struct L1Jacobi{F<:AbstractFloat} <: AbstractSmoother
    omega::F
end
Adapt.@adapt_structure L1Jacobi

L1Jacobi(; omega=1.0) = L1Jacobi{Float64}(Float64(omega))

# ─── User-facing AMG marker type ──────────────────────────────────────────────

"""
    AMG(; smoother, cycle, max_levels, coarsest_size, pre_sweeps, post_sweeps,
          strength, coarsening, update_freq)

Algebraic Multigrid linear solver for use with `SolverSetup`.

# Keyword arguments
- `smoother` — smoother at each level. `JacobiSmoother(; domain, loops, omega)` or
  `Chebyshev(; degree, lo, hi)`.
- `cycle` — multigrid cycle type: `VCycle()` (default) or `WCycle()`.
- `max_levels` — maximum number of grid levels (default 25).
- `coarsest_size` — stop coarsening when matrix size ≤ this threshold (default 50).
- `pre_sweeps` — number of pre-smoothing sweeps per level (default 2).
- `post_sweeps` — number of post-smoothing sweeps per level (default 2).
- `strength` — strength-of-connection threshold θ for coarsening (default 0.0).
  A connection (i,j) is kept when `|a_ij| ≥ θ · max_{k≠i} |a_ik|`. Smaller θ
  retains more connections (denser coarse levels, slower coarsening); larger θ
  keeps only the strongest connections (sparser coarse levels, more aggressive
  coarsening).

  **For FVM pressure (Laplacian) matrices the correct default is `θ = 0.0`.**
  These are near-isotropic M-matrices in which every off-diagonal entry
  contributes equally to the solution, so all connections should be treated as
  "strong." On non-uniform meshes (e.g. wall-refined cylinder grids) cells near
  the wall can have radial coefficients 4–16× larger than tangential ones.
  With `θ = 0.25` the tangential connections are dropped from the strong graph,
  near-wall cells aggregate only in the radial direction, the prolongation
  quality is poor, and AMG fails to converge — causing the outer
  PISO/SIMPLE loop to diverge and eventually crash.

  Use a non-zero θ only for strongly anisotropic problems (e.g. aligned
  diffusion with anisotropy ratio ≫ 4) where selective coarsening along the
  strong direction is intentional.  For `:RS` (Ruge–Stüben) the classical
  value θ = 0.25 is standard, but still incorrect for near-isotropic FVM
  operators.  When in doubt, keep `strength = 0.0`.
- `coarsening` — coarsening strategy: `:SA` Smoothed Aggregation (default) or `:RS`
  Ruge–Stüben. Both use **unsmoothed (piecewise-constant) prolongation** to keep
  operator complexity low (target < 2.0 for 3-D FVM meshes). Prolongation smoothing
  would inflate nnz(P) from 1 to ~stencil_width per row with no benefit for
  near-isotropic M-matrices where all connections are equally strong.
- `update_freq` — how often the coarse-level hierarchy (Galerkin products and
  coarsest LU) is refreshed when the fine-level matrix changes (default `1`).
- `krylov` — outer Krylov acceleration. `:cg` (default) wraps the V-cycle as a
  preconditioner inside Preconditioned Conjugate Gradient (PCG), which is optimal
  for the SPD pressure Laplacian: each PCG step costs one V-cycle plus two dot
  products, but convergence is O(√κ) vs O(κ) for Richardson. Use `:none` to
  revert to the plain Richardson (V-cycle) iteration.
- `fine_float` — float type for the fine-level system matrix and work vectors
  (default `Float64`). Must match the element type of the matrix passed to
  `solve_system!`. Stored for documentation and future use; the actual fine-level
  float type is currently always derived from `eltype(b)`.
- `coarse_float` — float type for all coarse-level matrices and work vectors
  (default `Float32`). Set to `Float32` to use half-bandwidth memory on the GPU;
  this is the main mixed-precision knob. Changing this requires matching
  `_tc_sparse_type`/`_tc_vec_type` methods to exist for the desired type.

  In a SIMPLE/PISO loop, `update!` is called once per outer iteration. With
  `update_freq = 1` (default) the full hierarchy is rebuilt every call.
  Setting `update_freq = N > 1` applies a **lazy refresh**: the fine-level
  diagonal D⁻¹ is always updated (cheap and accuracy-critical for the smoother),
  but the Galerkin products and coarsest LU are recomputed only on calls
  1, N+1, 2N+1, … This is safe because:
  - The outer iteration (SIMPLE/PISO) itself is an iterative correction loop,
    so slightly stale coarse-level operators cause at most a small increase in
    the number of AMG cycles needed per outer iteration.
  - As the simulation approaches convergence, the matrix changes very slowly
    and skipping intermediate refreshes has negligible accuracy impact.

  Recommended values:
  - `update_freq = 1`  — fully accurate, no approximation (default).
  - `update_freq = 2`  — mild savings; good for transient simulations with
    many timesteps where A changes little within a timestep.
  - `update_freq = 3–5` — larger savings; suitable for near-converged steady
    runs or when the outer solver uses many inner iterations per timestep.

# Example
```julia
solvers = (
    p = SolverSetup(
        solver      = AMG(smoother    = JacobiSmoother(; domain=mesh, loops=2, omega=2/3),
                          cycle       = VCycle(),
                          coarsening  = :SA,
                          max_levels  = 20,
                          update_freq = 2),   # refresh hierarchy every 2 outer iterations
        preconditioner = Jacobi(),   # ignored by AMG (self-contained); kept for API compat
        convergence = 1e-8,
        relax = 0.2,
        rtol = 1e-3,
        itmax = 20,
    ),
)
```
"""
struct AMG{S<:AbstractSmoother, C<:AMGCycle} <: AbstractLinearSolver
    smoother      :: S
    cycle         :: C
    max_levels    :: Int
    coarsest_size :: Int
    pre_sweeps    :: Int
    post_sweeps   :: Int
    strength      :: Float64
    coarsening    :: Symbol
    update_freq   :: Int   # refresh Galerkin hierarchy every N update! calls (1 = every call)
    krylov        :: Symbol  # :cg → PCG outer loop; :none → plain Richardson
    fine_float    :: DataType   # float type for the fine level (default Float64)
    coarse_float  :: DataType   # float type for all coarse levels (default Float32)
end

AMG(;
    smoother      = JacobiSmoother(2, 2/3, zeros(0)),
    cycle         = VCycle(),
    max_levels    = 25,
    coarsest_size = 50,
    pre_sweeps    = 2,
    post_sweeps   = 1,
    strength      = 0.0,
    coarsening    = :RS,
    update_freq   = 2,
    krylov        = :cg,
    fine_float    = Float64,
    coarse_float  = Float32,
) = AMG(smoother, cycle, max_levels, coarsest_size, pre_sweeps, post_sweeps,
        Float64(strength), coarsening, update_freq, krylov, fine_float, coarse_float)

# ─── Host-only per-level extras ───────────────────────────────────────────────

"""
    LevelExtras{Tv, CpuSpT}

Host-resident mutable state for one multigrid level.

- `P_cpu`, `R_cpu` : CPU copies of the transfer operators. `nothing` at the coarsest level.
- `A_cpu`          : writable CPU mirror of the fine-level A (non-coarsest); coarsest-level
                     source for LU rebuild.
- `lu_dense`       : pre-allocated dense matrix for the coarsest-level LU (coarsest only).
- `rho`            : Gershgorin upper bound on ρ(D⁻¹A); used for Chebyshev eigenvalue bounds.
- `lu_factor`      : in-place LU factorisation of `lu_dense` (coarsest level only).
- `lu_rhs`         : scratch vector for the LU back-solve (coarsest level only).
- `diag_ptr`       : device-resident nzval index of `A[i,i]` per row; branch-free Dinv rebuild.
- `AP_cpu`         : pre-allocated intermediate A*P matrix (CPU, non-coarsest levels).
- `Ac_cpu`         : pre-allocated Ac output scratch (CPU, non-coarsest levels).
- `cpu_tmps`       : compact thread-local Float accumulator for `_spgemm_nzval!`; sized
                     `max_nnz_per_row(C) × nthreads` — fits in L1 cache.
- `col_to_local`   : thread-local Int32 scatter map sized `ncols_B × nthreads`; maps each
                     column j of B to the local slot in `cpu_tmps` for the current row. Set
                     per-row and cleared after write-back; never allocated at the hot path.
- `r_Tc`           : Float32 boundary buffer; fine level only. Holds `cast(r_fine)` for
                     the Float64→Float32 restriction step. `nothing` at coarse/single levels.
- `tmp_Tc`         : Float32 boundary buffer; fine level only. Holds `P * x_coarse` in
                     Float32 before casting back to Float64. `nothing` at coarse/single levels.

`CpuSpT` is `SparseMatricesCSR.SparseMatrixCSR{1, Tv, Int}` where `Tv` matches the level's
float type (`Float64` for the fine level, `Float32` for coarse levels).

`TcVec` is the concrete device vector type for the boundary buffers:
- Fine level: `Vector{Float32}` (CPU) or `CuArray{Float32,...}` (GPU) — the actual buffer type.
- Coarse levels: `Nothing` — boundary buffers are unused; field type collapses to `Nothing`.

Using a concrete `TcVec` parameter makes `r_Tc` and `tmp_Tc` a 2-option union
(`Union{Nothing, TcVec}`) rather than an abstract `AbstractVector`, keeping the cycle
hot path fully type-stable.

The struct itself is never adapted to the device (`Adapt.adapt` returns it unchanged).
`diag_ptr` is device-resident; all other fields live on the host.
"""
mutable struct LevelExtras{Tv, TcVec, CpuSpT}
    P_cpu        :: Union{Nothing, CpuSpT}
    R_cpu        :: Union{Nothing, CpuSpT}
    A_cpu        :: Union{Nothing, CpuSpT}
    lu_dense     :: Union{Nothing, Matrix{Tv}}
    rho          :: Tv
    lu_factor    :: Union{Nothing, LinearAlgebra.LU{Tv, Matrix{Tv}, Vector{Int}}}
    lu_rhs       :: Vector{Tv}
    diag_ptr     :: Union{Nothing, AbstractVector{Int32}}
    AP_cpu       :: Union{Nothing, CpuSpT}
    Ac_cpu       :: Union{Nothing, CpuSpT}
    cpu_tmps     :: Union{Nothing, Matrix{Tv}}
    col_to_local :: Union{Nothing, Matrix{Int32}}
    # Mixed-precision boundary buffers; TcVec=Nothing for coarse/single levels.
    r_Tc         :: Union{Nothing, TcVec}
    tmp_Tc       :: Union{Nothing, TcVec}

    LevelExtras{Tv, TcVec, CpuSpT}() where {Tv, TcVec, CpuSpT} =
        new{Tv, TcVec, CpuSpT}(nothing, nothing, nothing, nothing, one(Tv), nothing, Tv[],
                                nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

# ─── Per-level storage ────────────────────────────────────────────────────────

"""
    MultigridLevel{Tv, AType, PType, Vec, ExtrasT}

Immutable, fully-parametric struct for one level of the AMG hierarchy.

- `A`      : system matrix (device-resident, type `AType`).
- `P`, `R` : prolongation / restriction operators (`PType = Union{Nothing, AType}`).
             `nothing` at the coarsest level; a concrete sparse matrix elsewhere.
- `Dinv`   : inverse diagonal D⁻¹ (device-resident, type `Vec`).
- `x`, `b`, `r`, `tmp` : pre-allocated work vectors (device-resident, type `Vec`).
             `r` holds the residual in cycle functions and the Chebyshev smoother.
             The fused Jacobi kernel updates `x` in-place without writing to `r`.
- `extras` : host-only mutable state (`ExtrasT = LevelExtras{Tv, CpuSpT}`).

`Adapt.@adapt_structure` adapts all device fields (`A`, `P`, `R`, work vectors)
to the target backend while leaving `extras` untouched — `Adapt.adapt` returns a
`LevelExtras` unchanged since no `adapt_structure` method is defined for it.

The outer constructor always fixes `PType = Union{Nothing, AType}` so that every
level in the hierarchy shares the same concrete `MultigridLevel` type and can be
stored in a plain `Vector{MultigridLevel{...}}` without any dynamic dispatch.
"""
struct MultigridLevel{Tv, AType, PType, Vec <: AbstractVector{Tv}, ExtrasT}
    A      :: AType
    P      :: PType
    R      :: PType
    Dinv   :: Vec
    x      :: Vec
    b      :: Vec
    r      :: Vec
    tmp    :: Vec
    extras :: ExtrasT
end

Adapt.@adapt_structure MultigridLevel

function MultigridLevel(A::AType, P, R,
                         Dinv::Vec, x::Vec, b::Vec, r::Vec, tmp::Vec,
                         extras::ExtrasT) where {AType, Vec<:AbstractVector, ExtrasT}
    Tv    = eltype(Vec)
    PType = Union{Nothing, AType}
    MultigridLevel{Tv, AType, PType, Vec, ExtrasT}(A, P, R, Dinv, x, b, r, tmp, extras)
end

# ─── Mixed-precision sparse/vector type mapping ───────────────────────────────
# Map a Float64 device sparse type → its Float32 equivalent.
# The CPU case is handled here; GPU extensions add methods for their types.
# Called at workspace construction time to determine LCType.

_tc_sparse_type(::Type{SparseXCSR{Bi,Tv,Ti,N}}) where {Bi,Tv,Ti,N} = SparseXCSR{Bi,Float32,Ti,N}
_tc_vec_type(::Type{Array{Tv,N}}) where {Tv,N} = Array{Float32,N}

# Fallback: tells the user to add a method in the relevant GPU extension.
_tc_sparse_type(::Type{AT}) where {AT} =
    error("No Float32 sparse type mapping for $(AT). Add _tc_sparse_type in the GPU extension.")
_tc_vec_type(::Type{VT}) where {VT} =
    error("No Float32 vector type mapping for $(VT). Add _tc_vec_type in the GPU extension.")

# ─── Workspace ────────────────────────────────────────────────────────────────

"""
    AMGWorkspace{LFType, LCType, Vec, Opts}

Holds the complete two-tier multigrid hierarchy and is stored in `phiEqn.solver` in
place of a Krylov.jl workspace. Exposes a `.x` field so the existing `_copy!` kernel
in `solve_system!` continues to work unchanged.

**Two-tier layout** (mixed-precision):
- `fine_level::Union{Nothing, LFType}` — the Float64 fine level (set by `amg_setup!`).
- `coarse_levels::Vector{LCType}` — Float32 coarse levels; empty before setup.

`LFType` and `LCType` are both fully-concrete `MultigridLevel` subtypes, determined at
construction time in `_workspace`. `LFType` uses `Tv = Float64` and holds Float32 P/R
operators (for the bandwidth-efficient fine→coarse boundary SpMVs). `LCType` uses
`Tv = Float32` for all fields.

This split eliminates dynamic dispatch in the cycle hot path: `vcycle_fine!` is
specialised on the concrete `LFType` and `Vector{LCType}`, and `vcycle_coarse!`
recurses on `Vector{LCType}` alone.
"""
mutable struct AMGWorkspace{LFType, LCType, Vec, Opts<:AMG}
    fine_level   :: Union{Nothing, LFType}
    coarse_levels:: Vector{LCType}
    x            :: Vec
    opts         :: Opts
    setup_valid  :: Bool
    setup_count  :: Int
    update_count :: Int   # counts update! calls; drives the lazy Galerkin refresh
    # PCG workspace — only used when opts.krylov === :cg.
    # x_pcg is the PCG iterate; separate from fine_level.x which the V-cycle overwrites.
    x_pcg        :: Vec
    p_cg         :: Vec
end
