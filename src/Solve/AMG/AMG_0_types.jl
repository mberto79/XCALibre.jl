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

l1-Jacobi smoother: scales update by `1/||a_i||_1` instead of `1/a_ii`.
For MPI AMG; in serial on isotropic problems, use `JacobiSmoother` instead.
"""
struct L1Jacobi{F<:AbstractFloat} <: AbstractSmoother
    omega::F
end
Adapt.@adapt_structure L1Jacobi

L1Jacobi(; omega=1.0) = L1Jacobi{Float64}(Float64(omega))

# ─── User-facing AMG marker type ──────────────────────────────────────────────

"""
    AMG(; smoother, cycle, max_levels, coarsest_size, pre_sweeps, post_sweeps,
          strength, coarsening, update_freq, krylov, fine_float, coarse_float, smooth_P, trunc_P,
          coarse_sweeps)

Algebraic Multigrid solver for use with `SolverSetup`.

- `smoother`: `JacobiSmoother`, `Chebyshev`, or `L1Jacobi`.
- `cycle`: `VCycle()` (default) or `WCycle()` (slower on GPU).
- `max_levels`: max grid levels (default 25).
- `coarsest_size`: stop coarsening below this size (default 50).
- `pre_sweeps`, `post_sweeps`: smoothing per level (defaults 2, 1).
- `coarse_sweeps`: Jacobi sweeps at coarsest level when no dense LU (default 50).
- `strength`: θ at fine level only; θ=0 at coarse levels (default 0.1).
- `coarsening`: `:RS` (Ruge–Stüben, default) or `:SA` (Smoothed Aggregation).
- `smooth_P`: apply one Jacobi step to prolongation (default true); reduces PCG iterations at cost of op_complexity.
- `trunc_P`: per-row drop threshold for P (default 0); effective range 0.05–0.3.
- `update_freq`: refresh coarse hierarchy every N outer iterations (default 2). Always updates fine D⁻¹.
- `krylov`: `:cg` (PCG, default) or `:none` (Richardson).
- `fine_float`, `coarse_float`: float types (default Float64, Float32).
"""
struct AMG{S<:AbstractSmoother, C<:AMGCycle} <: AbstractLinearSolver
    smoother      :: S
    cycle         :: C
    max_levels    :: Int
    coarsest_size :: Int
    pre_sweeps    :: Int
    post_sweeps   :: Int
    coarse_sweeps :: Int   # Jacobi sweeps at coarsest level when no dense LU available
    strength      :: Float64
    coarsening    :: Symbol
    update_freq   :: Int   # refresh Galerkin hierarchy every N update! calls (1 = every call)
    krylov        :: Symbol  # :cg → PCG outer loop; :none → plain Richardson
    fine_float    :: DataType   # float type for the fine level (default Float64)
    coarse_float  :: DataType   # float type for all coarse levels (default Float32)
    smooth_P      :: Bool     # apply one Jacobi step to tentative P (SA-AMG); default true
    trunc_P       :: Float64  # per-row drop threshold for P after smoothing (0 = disabled)
end

AMG(;
    smoother      = JacobiSmoother(2, 2/3, zeros(0)),
    cycle         = VCycle(),
    max_levels    = 25,
    coarsest_size = 50,
    pre_sweeps    = 2,
    post_sweeps   = 1,
    coarse_sweeps = 50,
    strength      = 0.1,
    coarsening    = :RS,
    update_freq   = 2,
    krylov        = :cg,
    fine_float    = Float64,
    coarse_float  = Float32,
    smooth_P      = true,
    trunc_P       = 0.0,
) = AMG(smoother, cycle, max_levels, coarsest_size, pre_sweeps, post_sweeps, coarse_sweeps,
        Float64(strength), coarsening, update_freq, krylov, fine_float, coarse_float,
        smooth_P, Float64(trunc_P))

# ─── Host-only per-level extras ───────────────────────────────────────────────

"""
    LevelExtras{Tv, TcVec, CpuSpT}

Host-resident mutable state for one multigrid level.
- CPU transfers: `P_cpu`, `R_cpu`, `A_cpu`, `AP_cpu`, `Ac_cpu`.
- Coarse solve: `lu_dense`, `lu_factor`, `lu_rhs` (coarsest only, n ≤ _MAX_DENSE_LU_N).
- Smoother: `rho` (Gershgorin bound for Chebyshev), `diag_ptr` (device nzval index).
- Galerkin scratch: `cpu_tmps`, `col_to_local` (compact L1 accumulators).
- Mixed-precision boundary: `r_Tc`, `tmp_Tc` (Float32 buffers; fine level only).
`TcVec` is the concrete device vector type (keeps cycle hot path type-stable).
`diag_ptr` is device-resident; all other fields live on host.
"""
mutable struct LevelExtras{Tv, TcVec, CpuSpT}
    P_cpu        :: Union{Nothing, CpuSpT}
    R_cpu        :: Union{Nothing, CpuSpT}
    A_cpu        :: Union{Nothing, CpuSpT}
    lu_dense     :: Any   # Matrix{Tv} for dense LU
    rho          :: Tv
    lu_factor    :: Any   # Union{Nothing, LU{Tv,Matrix,…}(CPU) or LU{Tv,CuMatrix,…}(GPU)}
    lu_rhs       :: Any   # device Vector{Tv} for dense LU
    diag_ptr     :: Union{Nothing, AbstractVector{Int32}}
    AP_cpu       :: Union{Nothing, CpuSpT}
    Ac_cpu       :: Union{Nothing, CpuSpT}
    AP_device    :: Any   # pre-allocated device A*P; nothing on CPU / unsmoothed levels
    A_f32_nzval  :: Any   # pre-allocated Float32 nzval buffer for A cast; nothing until first use
    cpu_tmps     :: Union{Nothing, Matrix{Tv}}
    col_to_local :: Union{Nothing, Matrix{Int32}}
    # Mixed-precision boundary buffers; TcVec=Nothing for coarse/single levels.
    r_Tc         :: Union{Nothing, TcVec}
    tmp_Tc       :: Union{Nothing, TcVec}
    # true when this level's P/R use smoothed (SA-style) prolongation.
    # The GPU RAP kernel assumes 1-nnz/row P; set this to fall back to CPU SpGEMM.
    smooth_P     :: Bool

    LevelExtras{Tv, TcVec, CpuSpT}() where {Tv, TcVec, CpuSpT} =
        new{Tv, TcVec, CpuSpT}(nothing, nothing, nothing, nothing, one(Tv), nothing, Tv[],
                                nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, false)
end

# ─── Per-level storage ────────────────────────────────────────────────────────

"""
    MultigridLevel{Tv, AType, PType, Vec, ExtrasT}

Immutable level container: matrix `A`, transfer ops `P`/`R` (nothing at coarsest),
inverse diagonal `Dinv`, work vectors `x, b, r, tmp`, and host-only `extras`.
All device fields adapt to target backend; `extras` stays on host (no adapt_structure method).
Fully parametric — all levels in hierarchy share concrete type for dispatch-free cycles.
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
# CPU default: Float64 → Float32 mapping. GPU extensions add device-specific methods.

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

Complete mixed-precision hierarchy: fine level (Float64 A, Float32 P/R), coarse levels (Float32).
Stored in `phiEqn.solver`; exposes `.x` for compatibility with `_copy!` kernel.
Two-tier split eliminates dynamic dispatch in cycle hot path.
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
    # r_pcg is the CG residual — kept separate from fine_level.r which the V-cycle clobbers,
    # eliminating the 2 amg_copy! calls (save + restore) per PCG iteration.
    x_pcg        :: Vec
    p_cg         :: Vec
    r_pcg        :: Vec
    # Diagnostics: accumulated iteration count and call count (reset by user via amg_reset_stats!).
    _pcg_iters   :: Int
    _solve_count :: Int
end
