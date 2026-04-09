export AMG, VCycle, WCycle, Chebyshev

# ─── Cycle markers ────────────────────────────────────────────────────────────

abstract type AMGCycle end
struct VCycle <: AMGCycle end
struct WCycle <: AMGCycle end

# ─── Smoother marker (Chebyshev polynomial smoother) ──────────────────────────

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

# ─── User-facing marker type (zero-field struct like Bicgstab, Cg, etc.) ──────

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
struct AMG{S<:AbstractSmoother,C<:AMGCycle} <: AbstractLinearSolver
    smoother::S
    cycle::C
    max_levels::Int
    coarsest_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    strength::Float64
    coarsening::Symbol
end

AMG(;
    smoother    = JacobiSmoother(2, 2/3, zeros(0)),  # dummy; user must supply domain
    cycle       = VCycle(),
    max_levels  = 25,
    coarsest_size = 50,
    pre_sweeps  = 2,
    post_sweeps = 2,
    strength    = 0.25,
    coarsening  = :SA,
) = AMG(smoother, cycle, max_levels, coarsest_size, pre_sweeps, post_sweeps,
        Float64(strength), coarsening)

# ─── Per-level storage ────────────────────────────────────────────────────────

"""
    MultigridLevel

Stores the system matrix, prolongation/restriction operators, and work vectors for
one level of the AMG hierarchy. All arrays are pre-allocated during setup.

`P` and `R` are typed as `Any` so that different levels can hold different concrete
sparse-matrix types (CPU vs GPU) without requiring re-parameterisation.
"""
mutable struct MultigridLevel{Tv, AType, Vec<:AbstractVector{Tv}}
    # System matrix for this level (level 1 wraps the original device matrix)
    A        :: AType
    # Prolongation P (fine→coarse) and restriction R = Pᵀ (coarse→fine).
    # Typed `Any` to allow different concrete sparse types across levels / backends.
    P        :: Any
    R        :: Any
    # CPU-resident copies of P and R (always SparseMatrixCSR on host).
    # Cached here to avoid a GPU→CPU round-trip during update!; on CPU backend these
    # are the same objects as P/R (parent() is zero-copy).
    P_cpu    :: Any
    R_cpu    :: Any
    # Inverse diagonal D⁻¹ (for Jacobi and Chebyshev spectral estimates)
    Dinv     :: Vec
    # Work vectors – allocated once, reused every cycle (no per-cycle allocations)
    x        :: Vec   # current iterate / correction
    b        :: Vec   # right-hand side
    r        :: Vec   # residual r = b - Ax
    tmp      :: Vec   # scratch (swap buffer for Jacobi / Chebyshev)
    # Chebyshev: estimated spectral radius of D⁻¹A
    rho      :: Base.RefValue{Tv}
    # Coarsest level: dense LU factorisation (CPU host)
    lu_factor :: Union{Nothing, LinearAlgebra.LU{Tv,Matrix{Tv},Vector{Int}}}
    lu_rhs    :: Union{Nothing, Vector{Tv}}
end

function MultigridLevel(A, P, R, Dinv::Vec, x::Vec, b::Vec, r::Vec, tmp::Vec) where {Vec}
    Tv = eltype(Dinv)
    MultigridLevel{Tv, typeof(A), Vec}(
        A, P, R, nothing, nothing, Dinv, x, b, r, tmp,
        Ref(zero(Tv)), nothing, nothing,
    )
end

# Adapt device-resident fields; P_cpu/R_cpu/lu_factor/lu_rhs stay on the host.
function Adapt.adapt_structure(to, L::MultigridLevel{Tv}) where {Tv}
    A    = Adapt.adapt(to, L.A)
    P    = Adapt.adapt(to, L.P)
    R    = Adapt.adapt(to, L.R)
    Dinv = Adapt.adapt(to, L.Dinv)
    x    = Adapt.adapt(to, L.x)
    b    = Adapt.adapt(to, L.b)
    r    = Adapt.adapt(to, L.r)
    tmp  = Adapt.adapt(to, L.tmp)
    MultigridLevel{Tv, typeof(A), typeof(x)}(
        A, P, R, L.P_cpu, L.R_cpu, Dinv, x, b, r, tmp,
        L.rho, L.lu_factor, L.lu_rhs,
    )
end

# ─── Workspace (replaces Krylov.jl workspace) ─────────────────────────────────

"""
    AMGWorkspace

Holds the complete multigrid hierarchy and is stored in `phiEqn.solver` in place
of a Krylov.jl workspace. Exposes a `.x` field so the existing `_copy!` kernel
in `solve_system!` continues to work unchanged.
"""
mutable struct AMGWorkspace{Vec, Opts<:AMG}
    levels       :: Vector{Any}   # Vector of MultigridLevel (Any avoids UnionAll issues)
    x            :: Vec           # top-level solution (alias for levels[1].x after setup)
    opts         :: Opts
    setup_valid  :: Bool          # false → full setup needed on next solve
    setup_count  :: Int           # number of times full setup ran (for diagnostics)
end
