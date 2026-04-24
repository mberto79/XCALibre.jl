export AMG, SmoothAggregation, RugeStuben, AMGJacobi, AMGChebyshev, AMGSymmetricGaussSeidel, AMGL1Jacobi
export AMGWorkspace, AMGHierarchy, AMGLevel, AMGTimingStats

abstract type AbstractAMGCoarsening end
abstract type AbstractAMGSmoother end

struct SmoothAggregation{F<:AbstractFloat,C} <: AbstractAMGCoarsening
    strength_threshold::F
    smoother_weight::F
    truncate_factor::F
    max_interp_entries::Int
    interpolation_passes::Int
    strength_measure::Symbol
    filter_weak_connections::Bool
    near_nullspace::C
end
function SmoothAggregation(;
    strength_threshold=0.25,
    smoother_weight=0.67,
    truncate_factor=0.0,
    max_interp_entries=4,
    interpolation_passes=2,
    strength_measure=:classical,
    filter_weak_connections=true,
    near_nullspace=nothing
)
    strength_measure ∈ (:classical, :symmetric) ||
        throw(ArgumentError("SmoothAggregation strength_measure must be :classical or :symmetric"))
    max_interp_entries >= 0 || throw(ArgumentError("SmoothAggregation max_interp_entries must be nonnegative"))
    interpolation_passes > 0 || throw(ArgumentError("SmoothAggregation interpolation_passes must be positive"))
    SmoothAggregation(
        float(strength_threshold),
        float(smoother_weight),
        float(truncate_factor),
        max_interp_entries,
        interpolation_passes,
        strength_measure,
        filter_weak_connections,
        near_nullspace
    )
end
Adapt.@adapt_structure SmoothAggregation

struct RugeStuben{F<:AbstractFloat} <: AbstractAMGCoarsening
    strength_threshold::F
    strength_measure::Symbol
end
function RugeStuben(; strength_threshold=0.25, strength_measure=:classical)
    strength_measure ∈ (:classical, :symmetric) ||
        throw(ArgumentError("RugeStuben strength_measure must be :classical or :symmetric"))
    RugeStuben(float(strength_threshold), strength_measure)
end
Adapt.@adapt_structure RugeStuben

struct AMGJacobi{F<:AbstractFloat} <: AbstractAMGSmoother
    omega::F
end
AMGJacobi(; omega=2 / 3) = AMGJacobi(float(omega))
Adapt.@adapt_structure AMGJacobi

struct AMGSymmetricGaussSeidel <: AbstractAMGSmoother end
Adapt.@adapt_structure AMGSymmetricGaussSeidel

struct AMGL1Jacobi{F<:AbstractFloat} <: AbstractAMGSmoother
    omega::F
end
AMGL1Jacobi(; omega=2 / 3) = AMGL1Jacobi(float(omega))
Adapt.@adapt_structure AMGL1Jacobi

struct AMGChebyshev{I<:Integer,F<:AbstractFloat} <: AbstractAMGSmoother
    degree::I
    lower_fraction::F
    power_iterations::I
end
AMGChebyshev(; degree=2, lower_fraction=0.2, power_iterations=4) =
    AMGChebyshev(degree, float(lower_fraction), power_iterations)
Adapt.@adapt_structure AMGChebyshev

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
    assume_fixed_pattern::Bool
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
    max_coarse_rows=256,
    adaptive_rebuild_factor=0.85,
    coarse_refresh_interval=4,
    numeric_refresh_rtol=0.05,
    assume_fixed_pattern=true
)
    mode ∈ (:solver, :cg) || throw(ArgumentError("AMG mode must be :solver or :cg"))
    cycle ∈ (:V, :W) || throw(ArgumentError("AMG cycle must be :V or :W"))
    smoothing_steps = isnothing(smoothing_steps) ? (smoother isa AMGChebyshev ? 1 : 10) : smoothing_steps
    presweeps = isnothing(presweeps) ? smoothing_steps : presweeps
    postsweeps = isnothing(postsweeps) ? smoothing_steps : postsweeps
    presweeps > 0 || throw(ArgumentError("AMG presweeps must be positive"))
    postsweeps > 0 || throw(ArgumentError("AMG postsweeps must be positive"))
    coarsening isa SmoothAggregation && coarsening.truncate_factor < 0 &&
        throw(ArgumentError("SmoothAggregation truncate_factor must be nonnegative"))
    coarsening isa SmoothAggregation && coarsening.max_interp_entries < 0 &&
        throw(ArgumentError("SmoothAggregation max_interp_entries must be nonnegative"))
    coarsening isa SmoothAggregation && coarsening.interpolation_passes <= 0 &&
        throw(ArgumentError("SmoothAggregation interpolation_passes must be positive"))
    coarse_refresh_interval > 0 || throw(ArgumentError("AMG coarse_refresh_interval must be positive"))
    numeric_refresh_rtol >= 0 || throw(ArgumentError("AMG numeric_refresh_rtol must be nonnegative"))
    AMG(
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
        float(numeric_refresh_rtol),
        assume_fixed_pattern
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

mutable struct AMGLevel{
    M,
    P,
    R,
    V1<:AbstractVector,
    V2<:AbstractVector,
    V3<:AbstractVector,
    V4<:AbstractVector,
    VI<:AbstractVector,
    VC<:AbstractVector,
    T
}
    A::M
    P::P
    R::R
    diagonal::V1
    inv_diagonal::V2
    l1_inv_diagonal::V2
    rhs::V3
    x::V4
    tmp::V4
    aggregate_ids::VI
    coarse_work::VC
    lambda_max::T
    level_id::Int
    coarse_solver::Any
end
Adapt.@adapt_structure AMGLevel

mutable struct AMGHierarchy{L}
    levels::L
    backend::Any
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
    finest_snapshot::Any
    cpu_workspace::Any
    cpu_rhs::Any
    cpu_x::Any
end
Adapt.@adapt_structure AMGHierarchy

mutable struct AMGWorkspace{V<:AbstractVector,T}
    hierarchy::Any
    timing::AMGTimingStats
    solution::V
    residual::V
    correction::V
    search::V
    preconditioned::V
    q::V
    iterations::Int
    last_residual_norm::T
    last_relative_residual::T
    converged::Bool
    residual_history::Vector{Float64}
end
Adapt.@adapt_structure AMGWorkspace

function _workspace(::AMG, b)
    x = similar(b)
    AMGWorkspace(
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
        zero(eltype(x)),
        false,
        Float64[]
    )
end
