export AMG, SmoothAggregation, RugeStuben, AMGJacobi, AMGChebyshev, AMGSymmetricGaussSeidel, AMGL1Jacobi
export AMGWorkspace, AMGHierarchy, AMGLevel

abstract type AbstractAMGCoarsening end
abstract type AbstractAMGSmoother end

struct SmoothAggregation{F<:AbstractFloat,C} <: AbstractAMGCoarsening
    strength_threshold::F
    smoother_weight::F
    near_nullspace::C
end
SmoothAggregation(; strength_threshold=0.25, smoother_weight=0.67, near_nullspace=nothing) =
    SmoothAggregation(float(strength_threshold), float(smoother_weight), near_nullspace)
Adapt.@adapt_structure SmoothAggregation

struct RugeStuben{F<:AbstractFloat} <: AbstractAMGCoarsening
    strength_threshold::F
end
RugeStuben(; strength_threshold=0.25) = RugeStuben(float(strength_threshold))
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
    numeric_refresh_rtol=0.05
)
    mode ∈ (:solver, :cg) || throw(ArgumentError("AMG mode must be :solver or :cg"))
    cycle ∈ (:V, :W) || throw(ArgumentError("AMG cycle must be :V or :W"))
    smoothing_steps = isnothing(smoothing_steps) ? (smoother isa AMGChebyshev ? 1 : 10) : smoothing_steps
    presweeps = isnothing(presweeps) ? smoothing_steps : presweeps
    postsweeps = isnothing(postsweeps) ? smoothing_steps : postsweeps
    presweeps > 0 || throw(ArgumentError("AMG presweeps must be positive"))
    postsweeps > 0 || throw(ArgumentError("AMG postsweeps must be positive"))
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
        float(numeric_refresh_rtol)
    )
end
Adapt.@adapt_structure AMG

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
    AMGWorkspace(
        nothing,
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
