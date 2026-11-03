export AMG, Sweeps, NativeAMG, Pairwise, SmoothedAggregation, RugeStuben
export Chebyshev, GaussSeidel, SymGaussSeidel, SOR, SSOR
export Direct, Smoothed, Iterative
export AMGWorkspace, AMGHierarchy, AMGLevel, AMGRAPPlanCPU

abstract type AbstractAMGCoarsening end
abstract type AbstractAMGSmoother end
abstract type AbstractAMGCoarseSolve end
abstract type AbstractAMGWorkspace end
abstract type AbstractAMGHierarchy end

struct MaterialisedAMG end
struct MatrixFreeAMG end

# AMG's standalone V-cycle iteration is now spelled `solver=Richardson()`
Base.@deprecate_binding AMGSolver Richardson

# NEW SECTION: coarsest-level solves

"""
    Direct(; max_rows=512)

Solve the coarsest level exactly. On a GPU the factor stays on the device while the coarsest level
has at most `max_rows` rows, and is solved on the host otherwise (`max_rows=0` always uses the
host). On a CPU backend `max_rows` has no effect.
"""
struct Direct <: AbstractAMGCoarseSolve
    max_rows::Int
end
Direct(; max_rows=512) = (max_rows >= 0 || throw(ArgumentError("Direct max_rows must be non-negative"));
    Direct(Int(max_rows)))

"""
    Smoothed(smoother=Chebyshev(degree=10); sweeps=1)

Approximate the coarsest solve on the device by a fixed number of smoother sweeps from zero, a
fixed linear operator, so it is valid inside `Cg()`. Suits a large coarsest level (big
`coarsest`) on a GPU, where a factorisation would not fit. On a CPU backend it is a `Direct()` solve.
"""
struct Smoothed{S} <: AbstractAMGCoarseSolve
    smoother::S
    sweeps::Int
end

"""
    Iterative(solver=Bicgstab(); rtol=1e-2, atol=0.0, itmax=50)

Solve the coarsest level on the device with a Jacobi-preconditioned Krylov solver (`Cg()` for
symmetric matrices, `Bicgstab()` for any). The result depends on the right-hand side, so it needs
`SolverSetup(solver=Richardson())`. On a CPU backend it is a `Direct()` solve.
"""
struct Iterative{S,F} <: AbstractAMGCoarseSolve
    solver::S
    rtol::F
    atol::F
    itmax::Int
end

function Iterative(solver=Bicgstab(); rtol=1e-2, atol=0.0, itmax=50)
    solver isa Union{Cg,Bicgstab} || throw(ArgumentError("Iterative coarse solve takes Cg() or Bicgstab()"))
    itmax > 0 || throw(ArgumentError("Iterative itmax must be positive"))
    rtol, atol = promote(float(rtol), float(atol))
    return Iterative(solver, rtol, atol, Int(itmax))
end

Adapt.@adapt_structure Direct
Adapt.@adapt_structure Smoothed
Adapt.@adapt_structure Iterative

# NEW SECTION: coarsening

"""
    Pairwise(merges=1)

Pairwise agglomeration: each cell is paired with the neighbour it is most strongly coupled to
(largest `|aᵢⱼ|`), `merges` times per level, so clusters hold about `2^merges` cells. Transfer
between levels is piecewise constant and coarse matrices are Galerkin products. Cheap to set up,
light on memory and robust for non-symmetric matrices; the default.
"""
struct Pairwise <: AbstractAMGCoarsening
    merges::Int
    function Pairwise(merges::Integer=1)
        merges > 0 || throw(ArgumentError("Pairwise merges must be positive"))
        return new(Int(merges))
    end
end

"""
    SmoothedAggregation(; threshold=0.0, weight=4/3, near_nullspace=nothing)

Smoothed-aggregation AMG: aggregates of strongly coupled cells (`|aᵢⱼ|² ≥ threshold²·|aᵢᵢ aⱼⱼ|`) with
a Jacobi-smoothed interpolation (`weight` scaled by the largest eigenvalue of D⁻¹A). Fewer
iterations than `Pairwise` on hard symmetric problems, at a higher setup and memory cost.
"""
struct SmoothedAggregation{F,C} <: AbstractAMGCoarsening
    threshold::F
    weight::F
    near_nullspace::C
end
SmoothedAggregation(; threshold=0.0, weight=4/3, near_nullspace=nothing) =
    SmoothedAggregation(promote(float(threshold), float(weight))..., near_nullspace)

"""
    RugeStuben(; threshold=0.25)

Classical (Ruge-Stüben) AMG with direct interpolation. Serial meshes only; on a distributed mesh use
`BoomerAMG()`.
"""
struct RugeStuben{F} <: AbstractAMGCoarsening
    threshold::F
end
function RugeStuben(; threshold=0.25, legacy...)
    isempty(legacy) || (_amg_deprecated("RugeStuben(strength_threshold)", "RugeStuben(threshold)");
        threshold = legacy[:strength_threshold])
    RugeStuben(float(threshold))
end

Adapt.@adapt_structure Pairwise
Adapt.@adapt_structure SmoothedAggregation
Adapt.@adapt_structure RugeStuben

# NEW SECTION: smoothers

"""
    Chebyshev(; degree=3, eig_ratio=10.0, lambda_scale=1.1)

Chebyshev polynomial smoother of the given `degree` on Jacobi-scaled levels, targeting the spectrum
`[λmax/eig_ratio, λmax]` (λmax estimated per level, inflated by `lambda_scale`). Runs on CPU and GPU.
"""
struct Chebyshev{F} <: AbstractAMGSmoother
    degree::Int
    eig_ratio::F
    lambda_scale::F
end

function Chebyshev(; degree=3, eig_ratio=10.0, lambda_scale=1.1)
    degree > 0 || throw(ArgumentError("Chebyshev degree must be positive"))
    eig_ratio > 1 || throw(ArgumentError("Chebyshev eig_ratio must be greater than one"))
    lambda_scale > 0 || throw(ArgumentError("Chebyshev lambda_scale must be positive"))
    return Chebyshev(Int(degree), promote(float(eig_ratio), float(lambda_scale))...)
end

"""
    GaussSeidel()
    SymGaussSeidel()

Forward Gauss-Seidel, or a forward then a backward sweep. CPU only.
"""
struct GaussSeidel <: AbstractAMGSmoother end
struct SymGaussSeidel <: AbstractAMGSmoother end

"""
    SOR(omega=1.0)
    SSOR(omega=1.0)

Successive over-relaxation with the direct factor `omega` in (0, 2), forward (`SOR`) or forward then
backward (`SSOR`); `omega=1` is `GaussSeidel()`/`SymGaussSeidel()`. CPU only.
"""
struct SOR{F} <: AbstractAMGSmoother
    omega::F
    SOR(omega::Real=1.0) = (0 < omega < 2 || throw(ArgumentError("SOR omega must lie in (0, 2)"));
        new{typeof(float(omega))}(float(omega)))
end
struct SSOR{F} <: AbstractAMGSmoother
    omega::F
    SSOR(omega::Real=1.0) = (0 < omega < 2 || throw(ArgumentError("SSOR omega must lie in (0, 2)"));
        new{typeof(float(omega))}(float(omega)))
end

Adapt.@adapt_structure Chebyshev
Adapt.@adapt_structure GaussSeidel
Adapt.@adapt_structure SymGaussSeidel
Adapt.@adapt_structure SOR
Adapt.@adapt_structure SSOR

const AMGSmoother = Union{Jacobi,Chebyshev,GaussSeidel,SymGaussSeidel,SOR,SSOR}
const SweepSmoother = Union{GaussSeidel,SymGaussSeidel,SOR,SSOR} # sequential, CPU only

# sweep order and relaxation factor of the sequential smoothers
struct _Forward end
struct _Symmetric end
_sweep_order(::Union{GaussSeidel,SOR}) = _Forward()
_sweep_order(::Union{SymGaussSeidel,SSOR}) = _Symmetric()
_sweep_omega(::Union{GaussSeidel,SymGaussSeidel}) = 1.0
_sweep_omega(s::Union{SOR,SSOR}) = s.omega

_smoother_name(s) = string(nameof(typeof(s)))

_check_smoother_backend(backend, smoother) = smoother
_check_smoother_backend(backend::KernelAbstractions.GPU, smoother::SweepSmoother) = throw(ArgumentError(
    "$(_smoother_name(smoother))() is a sequential CPU smoother and cannot run on $(nameof(typeof(backend)))(); " *
    "use Jacobi() or Chebyshev()"))

# NEW SECTION: sweeps

"""
    Sweeps(pre, post; grow=0, max=Base.max(pre, post, 4))

Smoother sweeps before (`pre`) and after (`post`) the coarse correction on each level: level `l`
(finest `l = 1`) runs `min(pre + grow*(l-1), max)` sweeps before and the
same rule for `post` after. `AMG(sweeps=2)` is `Sweeps(2, 2)` and `AMG(sweeps=(0, 2))` is
`Sweeps(0, 2)`. Unequal `pre` and `post` make the V-cycle non-symmetric, which `Cg()` tolerates
through its flexible update; a symmetric V-cycle remains the better partner for `Cg()`.
"""
struct Sweeps
    pre::Int
    post::Int
    grow::Int
    max::Int
end

function Sweeps(pre::Integer, post::Integer; grow=0, max=Base.max(pre, post, 4))
    (pre >= 0 && post >= 0 && pre + post > 0) ||
        throw(ArgumentError("Sweeps needs pre, post ≥ 0 and at least one sweep"))
    grow >= 0 || throw(ArgumentError("Sweeps grow must be non-negative"))
    max >= Base.max(pre, post) || throw(ArgumentError("Sweeps max must be at least pre and post"))
    return Sweeps(Int(pre), Int(post), Int(grow), Int(max))
end
Sweeps(n::Integer) = Sweeps(n, n)
Sweeps(t::Tuple{Integer,Integer}) = Sweeps(t...)
Sweeps(s::Sweeps) = s

_pre_sweeps(s::Sweeps, level) = min(s.pre + s.grow * (level - 1), s.max)
_post_sweeps(s::Sweeps, level) = min(s.post + s.grow * (level - 1), s.max)

# NEW SECTION: AMG

"""
    NativeAMG(; coarse=Direct(), scale_correction=true, fuse=0)

Settings of XCALibre's own AMG (serial meshes) that have no PETSc counterpart:

- `coarse`: coarsest-level solve, `Direct()`, `Smoothed(...)` or `Iterative(...)`
- `scale_correction`: scale each coarse correction by the energy-optimal factor `(r·c)/(c·Ac)`
- `fuse`: with `Pairwise()`, the number of coarse levels below the finest that are not stored but
  applied through the finer matrix; saves memory at the cost of extra matrix products per V-cycle.
  Needs `Jacobi()` or `Chebyshev()`
"""
struct NativeAMG{CS}
    coarse::CS
    scale_correction::Bool
    fuse::Int
end

function NativeAMG(; coarse=Direct(), scale_correction=true, fuse=0)
    coarse isa AbstractAMGCoarseSolve ||
        throw(ArgumentError("NativeAMG coarse must be Direct(), Smoothed(...) or Iterative(...)"))
    coarse isa Smoothed && _check_smoothed_coarse(coarse)
    fuse >= 0 || throw(ArgumentError("NativeAMG fuse must be non-negative"))
    return NativeAMG(coarse, Bool(scale_correction), Int(fuse))
end

Smoothed(smoother=Chebyshev(degree=10); sweeps=1) =
    (sweeps > 0 || throw(ArgumentError("Smoothed sweeps must be positive")); _check_smoothed_coarse(Smoothed(smoother, Int(sweeps))))
_check_smoothed_coarse(s::Smoothed{<:Union{Jacobi,Chebyshev}}) = s
_check_smoothed_coarse(s::Smoothed) = throw(ArgumentError("Smoothed coarse solve takes Jacobi() or Chebyshev()"))

Adapt.@adapt_structure NativeAMG

"""
    AMG(coarsening=Pairwise(); smoother=Jacobi(), sweeps, coarsest=4096, refresh=5, native=NativeAMG()) <: PreconditionerType

Algebraic multigrid preconditioner: `SolverSetup(solver=Cg(), preconditioner=AMG())` for symmetric
systems such as pressure, `solver=Bicgstab()` for non-symmetric ones (compressible pressure,
momentum), `solver=Richardson()` for plain V-cycles.

- `coarsening`: `Pairwise()`, `SmoothedAggregation()` or `RugeStuben()`
- `smoother`: `Jacobi()`, `Chebyshev()`, `GaussSeidel()`, `SymGaussSeidel()`, `SOR(ω)`, `SSOR(ω)`
- `sweeps`: `n`, `(pre, post)` or [`Sweeps`](@ref); defaults to 2 for `Jacobi()`, 1 otherwise
- `coarsest`: stop coarsening once a level has at most this many rows
- `refresh`: the coarse levels are recomputed from the current matrix every `refresh` solves; the
  aggregates and interpolation are kept while the sparsity pattern is unchanged
- `native`: [`NativeAMG`](@ref) settings used on serial meshes only

A serial mesh runs XCALibre's own AMG; a distributed mesh runs PETSc's GAMG with the same coarsening
family, smoother and sweeps (see the AMG section of the user guide for what differs). Settings a
path cannot honour stop the run there. `SolverSetup(precision=MixedF32())` stores the coarse levels
in Float32.
"""
struct AMG{C,S,N,TS} <: PreconditionerType
    coarsening::C
    smoother::S
    sweeps::Sweeps
    coarsest::Int
    refresh::Int
    native::N
end

function AMG(coarsening=Pairwise(); smoother=Jacobi(), sweeps=(smoother isa Jacobi ? 2 : 1),
    coarsest=4096, refresh=5, native=NativeAMG(), legacy...)
    isempty(legacy) || return _legacy_amg(coarsening, smoother, sweeps, coarsest, refresh, native, legacy)
    coarsening isa AbstractAMGCoarsening ||
        throw(ArgumentError("AMG coarsening must be Pairwise(), SmoothedAggregation() or RugeStuben()"))
    smoother isa AMGSmoother || throw(ArgumentError(
        "AMG smoother must be Jacobi(), Chebyshev(), GaussSeidel(), SymGaussSeidel(), SOR() or SSOR()"))
    native isa NativeAMG || throw(ArgumentError("AMG native must be NativeAMG(...)"))
    coarsest > 0 || throw(ArgumentError("AMG coarsest must be positive"))
    native.fuse == 0 || (coarsening isa Pairwise && smoother isa Union{Jacobi,Chebyshev}) || throw(ArgumentError(
        "NativeAMG(fuse) needs Pairwise() coarsening and the Jacobi() or Chebyshev() smoother"))
    refresh > 0 || throw(ArgumentError("AMG refresh must be positive"))
    return AMG{typeof(coarsening),typeof(smoother),typeof(native),Float64}(
        coarsening, smoother, Sweeps(sweeps), Int(coarsest), Int(refresh), native)
end

Adapt.adapt_structure(to, a::AMG) = _with_storage(AMG(adapt(to, a.coarsening); smoother=adapt(to, a.smoother),
    sweeps=a.sweeps, coarsest=a.coarsest, refresh=a.refresh, native=adapt(to, a.native)), _amg_storage(a))

# storage precision of the coarse levels (Float64 = that of the equation); MixedF32 sets Float32
_amg_storage(::AMG{C,S,N,TS}) where {C,S,N,TS} = TS
_with_storage(a::AMG{C,S,N}, ::Type{TS}) where {C,S,N,TS} =
    AMG{C,S,N,TS}(a.coarsening, a.smoother, a.sweeps, a.coarsest, a.refresh, a.native)

# Clamp to min(T,TS): storing hierarchy at higher precision than working matrix wastes memory
_effective_storage(::Type{T}, ::Type{TS}) where {T,TS} = sizeof(TS) >= sizeof(T) ? T : TS

# a safety limit on the level count; `coarsest` normally stops coarsening first
const _AMG_MAX_LEVELS = 50

# SuiteSparse rejects Float32 sparse factorizations; coarsest direct solve always runs Float64
_coarse_direct_eltype(::Type{Float32}) = Float64
_coarse_direct_eltype(::Type{T}) where {T} = T

# an iterative coarse solve depends on the rhs, which breaks the fixed preconditioner Krylov needs
_check_amg_solver(a::AMG, solver) = _check_amg_solver(a.native.coarse, solver)
_check_amg_solver(::Iterative, ::Union{Cg,Bicgstab}) = throw(ArgumentError(
    "NativeAMG(coarse=Iterative(...)) needs solver=Richardson(): an iterative coarse solve depends on the " *
    "right-hand side, which a Krylov preconditioner must not"))
_check_amg_solver(coarse, solver) = solver

_amg_solver_name(::Richardson) = "richardson"
_amg_solver_name(::Cg) = "cg"
_amg_solver_name(::Bicgstab) = "bicgstab"



struct AMGMatrixCSR{RP,CV,NZ}
    rowptr::RP
    colval::CV
    nzval::NZ
    m::Int
    n::Int
end

Adapt.@adapt_structure AMGMatrixCSR

Base.size(A::AMGMatrixCSR) = (A.m, A.n)
Base.size(A::AMGMatrixCSR, d::Integer) = d == 1 ? A.m : d == 2 ? A.n : 1
SparseArrays.nnz(A::AMGMatrixCSR) = length(A.nzval)
_m(A::AMGMatrixCSR) = A.m
_n(A::AMGMatrixCSR) = A.n
_rowptr(A::AMGMatrixCSR) = A.rowptr
_colval(A::AMGMatrixCSR) = A.colval
_nzval(A::AMGMatrixCSR) = A.nzval
_rowptr(A::SparseXCSR) = parent(A).rowptr
_colval(A::SparseXCSR) = parent(A).colval
_nzval(A::SparseXCSR) = parent(A).nzval
_rowptr(A::SparseMatricesCSR.SparseMatrixCSR) = A.rowptr
_colval(A::SparseMatricesCSR.SparseMatrixCSR) = A.colval
_nzval(A::SparseMatricesCSR.SparseMatrixCSR) = A.nzval

struct AMGGalerkinCache{I,W}
    targets::Vector{I}
    fine_indices::Vector{I}
    weights::Vector{W}
end

struct AMGRAPPlanCPU{I, T}
    ra_rowptr::Vector{I}
    ra_colval::Vector{I}
    ra_nzval::Vector{T}
    workspace_ra::Vector{T}
    workspace_rap::Vector{T}
    flag_ra::Vector{I}
    flag_rap::Vector{I}
end

mutable struct AMGLevel{MA,MP,MR,VD,VI,VX,T}
    A::MA
    P::MP
    R::MR
    diagonal::VD
    inv_diagonal::VD
    diagonal_index::VI
    rhs::VX
    x::VX
    tmp::VX
    direction::VX
    aggregate_ids::VI
    lambda_max::T
    level_id::Int
    has_transfer::Bool
end

Adapt.@adapt_structure AMGLevel

mutable struct AMGCPUCoarseLevel{MA,MC,MI,VX,LUF,QRF}
    A::MA
    Acsc::MC
    csc_nzval_index::MI
    rhs::VX
    x::VX
    lu_factor::LUF
    qr_factor::QRF
    use_qr::Bool
end

mutable struct AMGHierarchy{LD,LH,CC,B,RP,CP} <: AbstractAMGHierarchy
    levels::LD
    host_levels::LH
    coarse_cpu::CC
    backend::B
    workgroup::Int
    nrows::Int
    nnz::Int
    rowptr_pattern::RP
    colval_pattern::CP
    rowptr_ref::Base.RefValue{Any}
    colval_ref::Base.RefValue{Any}
    pattern_hash::UInt64
    transfer_csc::Vector{Any}
    galerkin_caches::Vector{Any}
    is_symmetric::Bool
    operator_complexity::Float64
    grid_complexity::Float64
    last_cycle_factor::Float64
    coarse_inv::Base.RefValue{Any}
    cycle_input::Base.RefValue{Any}
end

mutable struct AMGWorkspace{H,V,T,RH} <: AbstractAMGWorkspace
    hierarchy::H
    refresh_count::Int
    solution::V
    residual::V
    correction::V
    search::V
    preconditioned::V
    q::V
    # BiCGStab only, zero-length in other modes (see `_workspace`). None may alias:
    # `shadow` is held fixed for the whole solve, `t = A*M^-1*s` is a second matvec
    # target, and `svec` must not be `solution`, which `solve_system!` solves in place.
    shadow::V
    t::V
    svec::V
    iterations::Int
    converged::Bool
    last_relative_residual::T
    residual_history::RH
end

function _amg_backend_array(backend, values)
    return adapt(backend, values)
end

function _empty_amg_matrix(backend, ::Type{T}, ::Type{TI}=Int) where {T,TI}
    rowptr = KernelAbstractions.zeros(backend, TI, 1)
    colval = KernelAbstractions.zeros(backend, TI, 0)
    nzval = KernelAbstractions.zeros(backend, T, 0)
    return AMGMatrixCSR(rowptr, colval, nzval, 0, 0)
end

function _empty_amg_level(backend, ::Type{T}, ::Type{TI}=Int) where {T,TI}
    A = _empty_amg_matrix(backend, T, TI)
    P = _empty_amg_matrix(backend, T, TI)
    R = _empty_amg_matrix(backend, T, TI)
    diag = KernelAbstractions.zeros(backend, T, 0)
    invdiag = KernelAbstractions.zeros(backend, T, 0)
    diag_index = KernelAbstractions.zeros(backend, TI, 0)
    rhs = KernelAbstractions.zeros(backend, T, 0)
    x = KernelAbstractions.zeros(backend, T, 0)
    tmp = KernelAbstractions.zeros(backend, T, 0)
    direction = KernelAbstractions.zeros(backend, T, 0)
    aggregate_ids = KernelAbstractions.zeros(backend, TI, 0)
    return AMGLevel(A, P, R, diag, invdiag, diag_index, rhs, x, tmp, direction, aggregate_ids, zero(T), 0, false)
end

function _placeholder_lu_qr(::Type{T}) where {T}
    A = sparse([1], [1], [one(T)], 1, 1)
    return lu(A), qr(A)
end

function _empty_cpu_coarse_level(::Type{T}, ::Type{TI}=Int) where {T,TI}
    TC = _coarse_direct_eltype(T)
    lu_factor, qr_factor = _placeholder_lu_qr(TC)
    A = AMGMatrixCSR(TI[1, 1], TI[], T[], 1, 1)
    Acsc = sparse([1], [1], [one(TC)], 1, 1)
    csc_nzval_index = Int[]
    rhs = zeros(TC, 1)
    x = zeros(TC, 1)
    return AMGCPUCoarseLevel(A, Acsc, csc_nzval_index, rhs, x, lu_factor, qr_factor, false)
end

function _empty_hierarchy(backend, ::Type{T}, ::Type{TS}=T, ::Type{TI}=Int) where {T,TS,TI}
    host_level = _empty_amg_level(CPU(), T, TI)
    # Device level at TS so workspace.hierarchy reassignment after build type-checks
    device_level = _empty_amg_level(backend, TS, TI)
    host_levels = typeof(host_level)[]
    # GPU levels are heterogeneously typed (finest may be CuSparseMatrixCSR)
    device_levels = backend isa CPU ? typeof(device_level)[] : Vector{Any}()
    coarse_cpu = _empty_cpu_coarse_level(T, TI)
    return AMGHierarchy(
        device_levels,
        host_levels,
        coarse_cpu,
        backend,
        1,
        0,
        0,
        TI[],
        TI[],
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        UInt64(0),
        Any[],
        Any[],
        true,
        1.0,
        1.0,
        0.0,
        Ref{Any}(nothing),
        Ref{Any}(nothing)
    )
end

# Pairwise transfers are aggregate lists, never matrices; other coarsenings store P and R
amg_hierarchy_kind(a::AMG, backend) = a.coarsening isa Pairwise ? MatrixFreeAMG() : MaterialisedAMG()

_amg_empty_hierarchy(::MaterialisedAMG, backend, ::Type{T}, ::Type{TS}, ::Type{TI}, smoother) where {T,TS,TI} =
    _empty_hierarchy(backend, T, TS, TI)

# the hierarchy stores indices in the equation matrix index type, which the workspace type fixes here;
# `solver` is SolverSetup's outer iteration
function _workspace(amg::AMG, b, ::Type{TI}=Int; solver=Cg()) where {TI<:Integer}
    _check_amg_solver(amg, solver)
    T = eltype(b)
    TS = _effective_storage(T, _amg_storage(amg))
    backend = KernelAbstractions.get_backend(b)
    x = similar(b)
    # Only `amg_bicgstab_solve!` reads these three; allocating them in every mode adds
    # 50% to the workspace. `similar(x, 0)` keeps the field type concrete.
    bicg_vec() = solver isa Bicgstab ? similar(x) : similar(x, 0)
    return AMGWorkspace(
        _amg_empty_hierarchy(amg_hierarchy_kind(amg, backend), backend, T, TS, TI, amg.smoother),
        0,
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        bicg_vec(),   # shadow (BiCGStab)
        bicg_vec(),   # t      (BiCGStab)
        bicg_vec(),   # svec   (BiCGStab)
        0,
        false,
        zero(T),
        Float64[]
    )
end
