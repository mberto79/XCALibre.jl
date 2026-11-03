
export Preconditioner, PreconditionerType
export Jacobi, NormDiagonal #, ILU0 # , LDL
export DILU, DILUprecon
export BoomerAMG, GAMG
export IC0GPU, ILU0GPU

abstract type PreconditionerType end
abstract type LDIVPreconditioner <: PreconditionerType end
abstract type MULPreconditioner <: PreconditionerType end

struct NormDiagonal <: MULPreconditioner end
Adapt.@adapt_structure NormDiagonal

"""
    Jacobi(; weight=4/3)

Diagonal (Jacobi) preconditioner, and the Jacobi smoother of [`AMG`](@ref). As a smoother each sweep
relaxes by `weight/λmax`, with λmax the largest eigenvalue of D⁻¹A on that level, so the default
gives the usual 2/3 damping on Poisson-like levels (λmax ≈ 2); values are clamped below 2 for
stability. As a preconditioner `weight` plays no role.
"""
struct Jacobi <: MULPreconditioner
    weight::Float64
end
Jacobi(; weight=4/3) = (weight > 0 || throw(ArgumentError("Jacobi weight must be positive")); Jacobi(Float64(weight)))
Adapt.@adapt_structure Jacobi


struct DILU <: LDIVPreconditioner end
Adapt.@adapt_structure DILU

export HMIS, PMIS, Falgout, ExtI, Classical

abstract type HypreCoarsening end
abstract type HypreInterpolation end
struct HMIS <: HypreCoarsening end
struct PMIS <: HypreCoarsening end
struct Falgout <: HypreCoarsening end
struct ExtI <: HypreInterpolation end
struct Classical <: HypreInterpolation end

"""
    BoomerAMG(; threshold=0.7, coarsen=HMIS(), interpolation=ExtI(), max_interp=4, aggressive=1,
        smoother=SymGaussSeidel(), sweeps=1, refresh=10, device=false) <: PreconditionerType

hypre's classical AMG (BoomerAMG) through PETSc; it needs PETSc, so a serial mesh runs it after
`distribute(mesh)`. The stock Float64 PETSc libraries include hypre (Float32 ones do not). Use it
where [`AMG`](@ref) converges poorly or the strongest reduction per solve matters.

- `threshold`: strength of connection (0.5-0.7 in 3D)
- `coarsen`: `HMIS()`, `PMIS()` or `Falgout()`; `interpolation`: `ExtI()` or `Classical()`
- `max_interp`: interpolation entries per row; `aggressive`: levels of aggressive coarsening
- `smoother`: `SymGaussSeidel()` or `GaussSeidel()` (hybrid: Gauss-Seidel within each rank, Jacobi
  across ranks); `sweeps`: `n` or `(pre, post)`
- `refresh`: hypre cannot keep its interpolation, so the whole preconditioner is rebuilt from the
  current matrix every `refresh` solves; in between it is applied as built
- `device`: on GPU fields `BoomerAMG()` is refused unless PETSc's hypre reports device execution;
  `device=true` skips that check for a build the query misreads

The defaults keep operator complexity low in 3D. On a GPU with a CUDA-built hypre, PETSc switches to
PMIS coarsening and l1-Jacobi smoothing, so GPU and CPU residuals differ. Any other hypre option
goes through `petsc_options`.
"""
struct BoomerAMG{F,C<:HypreCoarsening,I<:HypreInterpolation,S} <: PreconditionerType
    threshold::F
    coarsen::C
    interpolation::I
    max_interp::Int
    aggressive::Int
    smoother::S
    sweeps::Tuple{Int,Int}
    refresh::Int
    device::Bool
end

function BoomerAMG(; threshold=0.7, coarsen=HMIS(), interpolation=ExtI(), max_interp=4, aggressive=1,
    smoother=SymGaussSeidel(), sweeps=1, refresh=10, device=false, legacy...)
    isempty(legacy) || throw(ArgumentError("BoomerAMG keywords are now typed (threshold, coarsen, interpolation, " *
        "max_interp, aggressive, smoother, sweeps, refresh); pass $(join(keys(legacy), ", ")) through " *
        "petsc_options=\"-pc_hypre_boomeramg_...\""))
    smoother isa Union{GaussSeidel,SymGaussSeidel} ||
        throw(ArgumentError("BoomerAMG smoother must be GaussSeidel() or SymGaussSeidel()"))
    s = sweeps isa Integer ? (sweeps, sweeps) : sweeps
    (all(>(0), s) && length(s) == 2) || throw(ArgumentError("BoomerAMG sweeps must be n > 0 or (pre, post) > 0"))
    refresh > 0 || throw(ArgumentError("BoomerAMG refresh must be positive"))
    return BoomerAMG(float(threshold), coarsen, interpolation, Int(max_interp), Int(aggressive), smoother,
        (Int(s[1]), Int(s[2])), Int(refresh), Bool(device))
end

"""
    GAMG(; freeze=25, agg_nsmooths=1, threshold=0.0)

Deprecated: use [`AMG`](@ref), which runs as PETSc's GAMG on a distributed mesh. `GAMG(...)` returns
the `AMG` of PETSc's GAMG defaults: `AMG(SmoothedAggregation(threshold=threshold), smoother=Chebyshev(degree=2,
eig_ratio=11), sweeps=1, coarsest=50, refresh=freeze)`, with `Pairwise()` for `agg_nsmooths=0`.
"""
function GAMG(; freeze=25, agg_nsmooths=1, threshold=0.0, reuse_interpolation=true, kwargs...)
    isempty(kwargs) || throw(ArgumentError("GAMG() is deprecated; use AMG(...) and pass $(join(keys(kwargs), ", ")) " *
        "with petsc_options=\"-pc_gamg_...\""))
    string(reuse_interpolation) == "true" || throw(ArgumentError("GAMG(reuse_interpolation=false) has no AMG equivalent"))
    string(agg_nsmooths) ∈ ("0", "1") || throw(ArgumentError("GAMG(agg_nsmooths) must be 0 or 1"))
    unsmoothed = string(agg_nsmooths) == "0"
    unsmoothed && threshold != 0 && throw(ArgumentError("GAMG(threshold) applies with agg_nsmooths=1 only"))
    coarsening = unsmoothed ? "Pairwise()" : "SmoothedAggregation(threshold=$threshold)"
    _LOG_RANK[]() && @warn "GAMG() is deprecated; use AMG(...). This GAMG() is AMG($coarsening, " *
        "smoother=Chebyshev(degree=2, eig_ratio=11), sweeps=1, coarsest=50, refresh=$freeze)" maxlog=1
    # PETSc's GAMG defaults: Chebyshev degree 2 on [0.1, 1.1]·λmax, coarsest at most 50 rows
    AMG(unsmoothed ? Pairwise() : SmoothedAggregation(; threshold); smoother=Chebyshev(degree=2, eig_ratio=11),
        sweeps=1, coarsest=50, refresh=freeze)
end

struct IC0GPU <: MULPreconditioner end
Adapt.@adapt_structure IC0GPU

struct ILU0GPU <: MULPreconditioner end
Adapt.@adapt_structure ILU0GPU

struct Preconditioner{T,M,P,S}
    A::M
    P::P
    storage::S
end
function Adapt.adapt_structure(to, itp::Preconditioner{T,M,Pr,S}) where {T,M,Pr,S}
    A = Adapt.adapt(to, itp.A)
    P = Adapt.adapt(to, itp.P)
    storage = Adapt.adapt(to, itp.storage) 
    Preconditioner{T,typeof(A),typeof(P),typeof(storage)}(A,P,storage)
end

is_ldiv(precon::Preconditioner{T,M,P,S}) where {T,M,P,S} = T <: LDIVPreconditioner

# held by an equation whose workspace (mixed precision, AMG) owns its preconditioner
struct InertPreconditioner <: PreconditionerType end
_inert_preconditioner() = Preconditioner{InertPreconditioner,Nothing,Nothing,Nothing}(nothing, nothing, nothing)
update_preconditioner!(::Preconditioner{InertPreconditioner}, mesh, config) = nothing

Preconditioner{NormDiagonal}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = KernelAbstractions.zeros(_active_backend(backend), F, m)
    P = diagonal_operator(S)
    Preconditioner{NormDiagonal,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{Jacobi}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = KernelAbstractions.zeros(_active_backend(backend), F, m)
    P = diagonal_operator(S)
    Preconditioner{Jacobi,typeof(A),typeof(P),typeof(S)}(A,P,S)
end


struct DILUprecon{M,V,VI}
    A::M
    D::V
    Di::VI
end
Adapt.@adapt_structure DILUprecon

Preconditioner{DILU}(A::SparseXCSR{N,F,I}) where {N,F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    Acsr = parent(A)
    D = zeros(F, m)
    Di = zeros(I, m)
    diagonal_indices!(Di, Acsr)
    S = DILUprecon(Acsr, D, Di)
    P = S
    Preconditioner{DILU,typeof(Acsr),typeof(P),typeof(S)}(Acsr,P,S)
end
