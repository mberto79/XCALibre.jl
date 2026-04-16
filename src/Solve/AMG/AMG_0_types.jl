export AMG, VCycle, WCycle, Chebyshev, L1Jacobi

# ─── Cycle markers ────────────────────────────────────────────────────────────

abstract type AMGCycle end
struct VCycle <: AMGCycle end
struct WCycle <: AMGCycle end

# ─── Chebyshev polynomial smoother ────────────────────────────────────────────

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
# Scales update by 1/||a_i||_1 instead of 1/a_ii; safer for anisotropic problems.

struct L1Jacobi{F<:AbstractFloat} <: AbstractSmoother
    omega::F
end
Adapt.@adapt_structure L1Jacobi

L1Jacobi(; omega=1.0) = L1Jacobi{Float64}(Float64(omega))

# ─── User-facing AMG marker type ──────────────────────────────────────────────

"""
    AMG(; smoother, cycle, coarsening, pre_sweeps, post_sweeps, ...)

Algebraic Multigrid preconditioned solver for use with `SolverSetup`.

# Key options
- `smoother`: `JacobiSmoother`, `Chebyshev`, or `L1Jacobi` (default: `JacobiSmoother`).
- `cycle`: `VCycle()` (default) or `WCycle()` (avoid on GPU — exponential coarse-level transfers).
- `coarsening`: `:SA` (Smoothed Aggregation, default) or `:RS` (Ruge–Stüben).
- `pre_sweeps`, `post_sweeps`: smoothing sweeps per level (default 2, 2).
- `coarsest_size`: stop coarsening at this level size (default 50000).
- `direct_solve_size`: coarsest-level size below which dense LU is used; above this Jacobi sweeps are used (default 20).
- `smooth_P`: Jacobi-smooth tentative prolongation (default `true`); fewer PCG iters, higher op_complexity.
- `strength`: off-diagonal strength threshold θ at fine level (default 0.1).
- `trunc_P`: per-row drop threshold for P after smoothing (default 0; effective range 0.05–0.3).
- `update_freq`: Galerkin hierarchy refresh interval in outer iterations (default 2); fine D⁻¹ always refreshed.
- `krylov`: `:cg` (PCG, default) or `:none` (Richardson).
- `fine_float`, `coarse_float`: float precision per tier (default `Float64`, `Float32`).
"""
struct AMG{S<:AbstractSmoother, C<:AMGCycle} <: AbstractLinearSolver
    smoother          :: S
    cycle             :: C
    max_levels        :: Int
    coarsest_size     :: Int
    direct_solve_size :: Int
    pre_sweeps        :: Int
    post_sweeps       :: Int
    coarse_sweeps     :: Int
    strength          :: Float64
    coarsening        :: Symbol
    update_freq       :: Int
    krylov            :: Symbol
    fine_float        :: DataType
    coarse_float      :: DataType
    smooth_P          :: Bool
    trunc_P           :: Float64
end

AMG(;
    smoother          = JacobiSmoother(2, 2/3, zeros(0)),
    cycle             = VCycle(),
    max_levels        = 15,
    coarsest_size     = 50000,
    direct_solve_size = 20,
    pre_sweeps        = 2,
    post_sweeps       = 2,
    coarse_sweeps     = 50,
    strength          = 0.1,
    coarsening        = :SA,
    update_freq       = 2,
    krylov            = :cg,
    fine_float        = Float64,
    coarse_float      = Float32,
    smooth_P          = true,
    trunc_P           = 0.0,
) = AMG(smoother, cycle, max_levels, coarsest_size, direct_solve_size, pre_sweeps, post_sweeps,
        coarse_sweeps, Float64(strength), coarsening, update_freq, krylov, fine_float,
        coarse_float, smooth_P, Float64(trunc_P))

# ─── Host-only per-level extras ───────────────────────────────────────────────

# diag_ptr is device-resident; lu_* only on coarsest level (n ≤ _MAX_DENSE_LU_N); r_Tc/tmp_Tc fine level only.
# TcVec=Nothing on coarse levels (no precision-boundary buffers needed).
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
    AP_device    :: Any   # pre-allocated device A*P (smooth_P path only)
    A_f32_nzval  :: Any   # Float32 nzval shadow (GPU precision-cast; nothing until first use)
    cpu_tmps     :: Union{Nothing, Matrix{Tv}}
    col_to_local :: Union{Nothing, Matrix{Int32}}
    r_Tc         :: Union{Nothing, TcVec}
    tmp_Tc       :: Union{Nothing, TcVec}
    smooth_P     :: Bool  # true → GPU uses multi-nnz RAP kernel; false → 1-nnz/row fast path

    LevelExtras{Tv, TcVec, CpuSpT}() where {Tv, TcVec, CpuSpT} =
        new{Tv, TcVec, CpuSpT}(nothing, nothing, nothing, nothing, one(Tv), nothing, Tv[],
                                nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, false)
end

# ─── Per-level storage ────────────────────────────────────────────────────────

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
# CPU default. GPU extensions add device-specific methods.

_tc_sparse_type(::Type{SparseXCSR{Bi,Tv,Ti,N}}) where {Bi,Tv,Ti,N} = SparseXCSR{Bi,Float32,Ti,N}
_tc_vec_type(::Type{Array{Tv,N}}) where {Tv,N} = Array{Float32,N}

# Fallback: tells the user to add a method in the relevant GPU extension.
_tc_sparse_type(::Type{AT}) where {AT} =
    error("No Float32 sparse type mapping for $(AT). Add _tc_sparse_type in the GPU extension.")
_tc_vec_type(::Type{VT}) where {VT} =
    error("No Float32 vector type mapping for $(VT). Add _tc_vec_type in the GPU extension.")

# ─── Workspace ────────────────────────────────────────────────────────────────

mutable struct AMGWorkspace{LFType, LCType, Vec, Opts<:AMG}
    fine_level   :: Union{Nothing, LFType}
    coarse_levels:: Vector{LCType}
    x            :: Vec
    opts         :: Opts
    setup_valid  :: Bool
    setup_count  :: Int
    update_count :: Int
    # PCG workspace (krylov === :cg only). x_pcg/r_pcg are separate from fine_level.x/r
    # so the V-cycle cannot clobber the CG iterate or residual between iterations.
    x_pcg        :: Vec
    p_cg         :: Vec
    r_pcg        :: Vec
    _pcg_iters   :: Int
    _solve_count :: Int
end
