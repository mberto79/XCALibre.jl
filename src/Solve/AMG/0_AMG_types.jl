export AMG, SmoothAggregation, RugeStuben, AMGJacobi
export AMGWorkspace, AMGHierarchy, AMGLevel, AMGTimingStats

abstract type AbstractAMGCoarsening end
abstract type AbstractAMGSmoother end
abstract type AbstractAMGWorkspace end

struct SmoothAggregation{F<:AbstractFloat,C,L,D} <: AbstractAMGCoarsening
    strength_threshold::F
    level_strength_thresholds::L
    smoother_weight::F
    near_nullspace::C
    max_prolongation_entries::Int
    aggressive_levels::Int
    aggressive_passes::Int
    coarse_drop_tolerances::D
end

function SmoothAggregation(;
    strength_threshold=0.10,
    level_strength_thresholds=(0.10, 0.075, 0.05),
    smoother_weight=1.0,
    near_nullspace=nothing,
    max_prolongation_entries=2,
    aggressive_levels=1,
    aggressive_passes=1,
    coarse_drop_tolerances=(0.0, 0.01, 0.03, 0.05)
)
    aggressive_levels >= 0 || throw(ArgumentError("SmoothAggregation aggressive_levels must be nonnegative"))
    aggressive_passes > 0 || throw(ArgumentError("SmoothAggregation aggressive_passes must be positive"))
    thresholds = isnothing(level_strength_thresholds) ? nothing : float.(collect(level_strength_thresholds))
    drop_tolerances = float.(collect(coarse_drop_tolerances))
    any(<(0), drop_tolerances) && throw(ArgumentError("SmoothAggregation coarse_drop_tolerances must be nonnegative"))
    return SmoothAggregation(
        float(strength_threshold),
        thresholds,
        float(smoother_weight),
        near_nullspace,
        max_prolongation_entries,
        aggressive_levels,
        aggressive_passes,
        drop_tolerances
    )
end

Adapt.@adapt_structure SmoothAggregation

struct RugeStuben{F<:AbstractFloat} <: AbstractAMGCoarsening
    strength_threshold::F
end

RugeStuben(; strength_threshold=0.05) = RugeStuben(float(strength_threshold))
Adapt.@adapt_structure RugeStuben

struct AMGJacobi{F<:AbstractFloat} <: AbstractAMGSmoother
    omega::F
end

AMGJacobi(; omega=0.6667) = AMGJacobi(float(omega))
Adapt.@adapt_structure AMGJacobi

struct AMG{C<:AbstractAMGCoarsening,S<:AbstractAMGSmoother} <: AbstractLinearSolver
    mode::Symbol
    coarsening::C
    smoother::S
    cycle::Symbol
    presweeps::Int
    postsweeps::Int
    max_levels::Int
    min_coarse_rows::Int
    max_coarse_rows::Int
    adaptive_rebuild_factor::Float64
    coarse_refresh_interval::Int
    numeric_refresh_rtol::Float64
end

function AMG(;
    mode=:solver,
    coarsening=SmoothAggregation(),
    smoother=AMGJacobi(),
    cycle=:V,
    smoothing_steps=nothing,
    presweeps=smoothing_steps,
    postsweeps=smoothing_steps,
    max_levels=10,
    min_coarse_rows=32,
    max_coarse_rows=512,
    adaptive_rebuild_factor=0.85,
    coarse_refresh_interval=mode == :cg ? typemax(Int) : 4,
    numeric_refresh_rtol=mode == :cg ? Inf : 0.05
)
    mode in (:solver, :cg) || throw(ArgumentError("AMG mode must be :solver or :cg"))
    coarsening isa Union{SmoothAggregation,RugeStuben} || throw(ArgumentError("AMG only supports SmoothAggregation() or RugeStuben() coarsening"))
    smoother isa AMGJacobi || throw(ArgumentError("AMG only supports AMGJacobi() smoothing"))
    cycle == :V || throw(ArgumentError("AMG only supports cycle=:V"))
    default_smoothing_steps = mode == :cg ? 2 : 1
    smoothing_steps = isnothing(smoothing_steps) ? default_smoothing_steps : smoothing_steps
    presweeps = isnothing(presweeps) ? smoothing_steps : presweeps
    postsweeps = isnothing(postsweeps) ? smoothing_steps : postsweeps
    presweeps > 0 || throw(ArgumentError("AMG presweeps must be positive"))
    postsweeps > 0 || throw(ArgumentError("AMG postsweeps must be positive"))
    coarse_refresh_interval > 0 || throw(ArgumentError("AMG coarse_refresh_interval must be positive"))
    numeric_refresh_rtol >= 0 || throw(ArgumentError("AMG numeric_refresh_rtol must be nonnegative"))
    return AMG(
        mode,
        coarsening,
        smoother,
        cycle,
        presweeps,
        postsweeps,
        max_levels,
        min_coarse_rows,
        max_coarse_rows,
        float(adaptive_rebuild_factor),
        coarse_refresh_interval,
        float(numeric_refresh_rtol)
    )
end

Adapt.@adapt_structure AMG

mutable struct AMGTimingStats
    build_time_s::Float64
    build_calls::Int
    refresh_time_s::Float64
    refresh_calls::Int
    finest_refresh_time_s::Float64
    finest_refresh_calls::Int
    apply_time_s::Float64
    apply_calls::Int
    last_update_action::Symbol
end

AMGTimingStats() = AMGTimingStats(0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, :none)

const AMGDirectFactorization{T,I} = Union{
    SparseArrays.UMFPACK.UmfpackLU{T,I},
    SparseArrays.SPQR.QRSparse{T,I}
}

struct AMGCoarseSolver{T,I<:Integer}
    factorization::AMGDirectFactorization{T,I}
end

const AMGTransferMatrix{T} = Union{
    Nothing,
    SparseMatrixCSC{T,Int},
    Transpose{T,SparseMatrixCSC{T,Int}}
}

mutable struct AMGLevel{
    M,
    V1<:AbstractVector,
    V2<:AbstractVector,
    V3<:AbstractVector,
    VI<:AbstractVector,
    VC<:AbstractVector,
    T
}
    A::M
    P::AMGTransferMatrix{T}
    R::AMGTransferMatrix{T}
    diagonal::V1
    inv_diagonal::V1
    rhs::V2
    x::V3
    tmp::V3
    aggregate_ids::VI
    coarse_work::VC
    lambda_max::T
    level_id::Int
    coarse_solver::Union{Nothing,AMGCoarseSolver{T,Int}}
end

Adapt.@adapt_structure AMGLevel

mutable struct AMGHierarchy{L,B,T}
    levels::L
    backend::B
    nrows::Int
    nnz::Int
    rowptr_pattern::Vector{Int}
    colval_pattern::Vector{Int}
    is_symmetric::Bool
    operator_complexity::Float64
    grid_complexity::Float64
    last_cycle_factor::Float64
    force_rebuild::Bool
    reuse_steps::Int
    finest_snapshot::Union{Nothing,Vector{T}}
    cpu_workspace::Union{Nothing,AbstractAMGWorkspace}
    cpu_rhs::Union{Nothing,Vector{T}}
    cpu_x::Union{Nothing,Vector{T}}
end

Adapt.@adapt_structure AMGHierarchy

mutable struct AMGWorkspace{V<:AbstractVector,T} <: AbstractAMGWorkspace
    hierarchy::Union{Nothing,AMGHierarchy}
    timing::AMGTimingStats
    solution::V
    residual::V
    correction::V
    search::V
    preconditioned::V
    q::V
    iterations::Int
    last_relative_residual::T
    residual_history::Vector{Float64}
end

Adapt.@adapt_structure AMGWorkspace

function _workspace(::AMG, b)
    x = similar(b)
    return AMGWorkspace(
        nothing,
        AMGTimingStats(),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        0,
        zero(eltype(x)),
        Float64[]
    )
end
