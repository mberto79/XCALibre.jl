export AMG, VCycle, WCycle, Chebyshev

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

# ─── User-facing AMG marker type ──────────────────────────────────────────────

"""
    AMG(; smoother, cycle, max_levels, coarsest_size, pre_sweeps, post_sweeps,
          strength, coarsening)

Algebraic Multigrid linear solver for use with `SolverSetup`.

# Keyword arguments
- `smoother` — smoother at each level. `JacobiSmoother(; domain, loops, omega)` or
  `Chebyshev(; degree, lo, hi)`.
- `cycle` — multigrid cycle type: `VCycle()` (default) or `WCycle()`.
- `max_levels` — maximum number of grid levels (default 25).
- `coarsest_size` — stop coarsening when matrix size ≤ this threshold (default 50).
- `pre_sweeps` — number of pre-smoothing sweeps per level (default 2).
- `post_sweeps` — number of post-smoothing sweeps per level (default 2).
- `strength` — strength-of-connection threshold θ for coarsening (default 0.25).
- `coarsening` — coarsening strategy: `:SA` Smoothed Aggregation (default) or `:RS`
  Ruge–Stüben.

# Example
```julia
solvers = (
    p = SolverSetup(
        solver      = AMG(smoother  = JacobiSmoother(; domain=mesh, loops=2, omega=2/3),
                          cycle     = VCycle(),
                          coarsening = :SA,
                          max_levels = 20),
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
end

AMG(;
    smoother      = JacobiSmoother(2, 2/3, zeros(0)),
    cycle         = VCycle(),
    max_levels    = 25,
    coarsest_size = 50,
    pre_sweeps    = 2,
    post_sweeps   = 2,
    strength      = 0.25,
    coarsening    = :SA,
) = AMG(smoother, cycle, max_levels, coarsest_size, pre_sweeps, post_sweeps,
        Float64(strength), coarsening)

# ─── Galerkin plan (device-resident) ─────────────────────────────────────────

"""
    GalerkinPlan{Vi}

Pre-computed index structure for the fused Galerkin product Ac = R·A·P.
Built once at `amg_setup!` time from the CPU CSR matrices; stored on the
target device so that `update!()` can compute all Ac nonzero values via a
single KA kernel launch — no CPU↔device transfers needed.

For each output nonzero `out` (1-based, into Ac.nzval):
    Ac.nzval[out] = Σ_p  R.nzval[nzi_R[p]] · A.nzval[nzi_A[p]] · P.nzval[nzi_P[p]]
where p ranges over `rowptr[out] : rowptr[out+1]-1`.

`Vi` is a device-resident integer vector (`Vector{Int32}` on CPU,
`CuVector{Int32}` on CUDA, etc.). `Adapt.@adapt_structure` moves the plan
to the target backend alongside the rest of the level data.
"""
struct GalerkinPlan{Vi <: AbstractVector{Int32}}
    rowptr :: Vi   # length = nnz(Ac) + 1; 1-based CSR row pointers into nzi_* arrays
    nzi_R  :: Vi   # 1-based index into R.nzval for each contributing triple
    nzi_A  :: Vi   # 1-based index into A.nzval for each contributing triple
    nzi_P  :: Vi   # 1-based index into P.nzval for each contributing triple
end
Adapt.@adapt_structure GalerkinPlan

# ─── Host-only per-level extras ───────────────────────────────────────────────

"""
    LevelExtras{Tv, CpuSpT}

Host-resident mutable state for one multigrid level. Never transferred to the GPU.

- `P_cpu`, `R_cpu` : CPU copies of the transfer operators (for `update!` Galerkin
  products without a GPU→CPU round-trip). `nothing` at the coarsest level.
- `rho`            : spectral radius estimate of D⁻¹A (for Chebyshev smoothing).
- `lu_factor`      : dense LU factorisation (coarsest level only; `nothing` otherwise).
- `lu_rhs`         : scratch vector for the LU back-solve (empty at non-coarsest levels).

`CpuSpT` is the concrete CPU sparse matrix type, always
`SparseMatricesCSR.SparseMatrixCSR{1, Tv, Int}` (produced by `sparsecsr`).

Since `LevelExtras` has no `Adapt.adapt_structure` method, `Adapt.adapt` returns it
unchanged — host-only state always stays on the CPU.
"""
mutable struct LevelExtras{Tv, CpuSpT}
    P_cpu         :: Union{Nothing, CpuSpT}
    R_cpu         :: Union{Nothing, CpuSpT}
    # CPU copy of the coarse matrix — stored only at the coarsest level so that
    # update!() can rebuild the LU factorisation by downloading just the nzval
    # from the device (rowptr/colval are fixed after setup).
    A_cpu         :: Union{Nothing, CpuSpT}
    # Device-resident plan for the fused KA Galerkin kernel (R·A·P).
    # Built once at setup; used every update!() call with zero CPU↔device transfers.
    galerkin_plan :: Union{Nothing, GalerkinPlan}
    lu_dense      :: Union{Nothing, Matrix{Tv}}  # pre-allocated dense matrix for coarsest LU
    rho           :: Tv
    lu_factor     :: Union{Nothing, LinearAlgebra.LU{Tv, Matrix{Tv}, Vector{Int}}}
    lu_rhs        :: Vector{Tv}

    LevelExtras{Tv, CpuSpT}() where {Tv, CpuSpT} =
        new{Tv, CpuSpT}(nothing, nothing, nothing, nothing, nothing, one(Tv), nothing, Tv[])
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
             The Jacobi smoother uses `r` as scratch for the correction
             (correction-form: `x += ω Dinv r` where `r = b - Ax`), so no
             field-swap is needed and the struct stays immutable.
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

# Outer constructor: fixes PType = Union{Nothing, AType} for a homogeneous hierarchy.
function MultigridLevel(A::AType, P, R,
                         Dinv::Vec, x::Vec, b::Vec, r::Vec, tmp::Vec,
                         extras::ExtrasT) where {AType, Vec<:AbstractVector, ExtrasT}
    Tv    = eltype(Vec)
    PType = Union{Nothing, AType}
    MultigridLevel{Tv, AType, PType, Vec, ExtrasT}(A, P, R, Dinv, x, b, r, tmp, extras)
end

# ─── Workspace ────────────────────────────────────────────────────────────────

"""
    AMGWorkspace{LType, Vec, Opts}

Holds the complete multigrid hierarchy and is stored in `phiEqn.solver` in place of
a Krylov.jl workspace. Exposes a `.x` field so the existing `_copy!` kernel in
`solve_system!` continues to work unchanged.

`LType` is the fully-concrete `MultigridLevel` type determined at construction time
from the matrix and RHS vector types via `_workspace(amg, A, b)`. The `levels` field
is therefore a fully-typed `Vector{LType}` — no `Any`, no dynamic dispatch in the
cycle hot path.
"""
mutable struct AMGWorkspace{LType, Vec, Opts<:AMG}
    levels      :: Vector{LType}
    x           :: Vec
    opts        :: Opts
    setup_valid :: Bool
    setup_count :: Int
end
