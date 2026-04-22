export AMG, SmoothAggregation, RugeStuben, AMGJacobi, AMGChebyshev
export AMGWorkspace, AMGHierarchy, AMGLevel

abstract type AbstractAMGCoarsening end
abstract type AbstractAMGSmoother end

struct SmoothAggregation{F<:AbstractFloat} <: AbstractAMGCoarsening
    strength_threshold::F
    smoother_weight::F
end
SmoothAggregation(; strength_threshold=0.25, smoother_weight=0.67) =
    SmoothAggregation(float(strength_threshold), float(smoother_weight))
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
    presweeps::Int
    postsweeps::Int
    max_levels::Int
    min_coarse_rows::Int
    max_coarse_rows::Int
end
function AMG(;
    mode=:solver,
    coarsening=SmoothAggregation(),
    smoother=AMGJacobi(),
    smoothing_steps=10,
    presweeps=smoothing_steps,
    postsweeps=smoothing_steps,
    max_levels=10,
    min_coarse_rows=32,
    max_coarse_rows=256
)
    mode ∈ (:solver, :cg) || throw(ArgumentError("AMG mode must be :solver or :cg"))
    presweeps > 0 || throw(ArgumentError("AMG presweeps must be positive"))
    postsweeps > 0 || throw(ArgumentError("AMG postsweeps must be positive"))
    AMG(mode, coarsening, smoother, presweeps, postsweeps, max_levels, min_coarse_rows, max_coarse_rows)
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
end
Adapt.@adapt_structure AMGHierarchy

mutable struct AMGWorkspace{V<:AbstractVector,T}
    hierarchy::Any
    residual::V
    correction::V
    search::V
    preconditioned::V
    q::V
    iterations::Int
    last_relative_residual::T
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
        0,
        zero(eltype(x))
    )
end
