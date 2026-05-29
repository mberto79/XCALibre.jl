export AMG, AMGSolver, VCycle, SmoothAggregation, RugeStuben, AMGJacobi, AMGChebyshev
export AMGWorkspace, AMGHierarchy, AMGLevel, AMGTimingStats

abstract type AbstractAMGMode end
abstract type AbstractAMGCoarsening end
abstract type AbstractAMGSmoother end
abstract type AbstractAMGCycle end
abstract type AbstractAMGWorkspace end

struct AMGSolver <: AbstractAMGMode end
struct VCycle <: AbstractAMGCycle end

struct SmoothAggregation{F,C,L,I,D,S} <: AbstractAMGCoarsening
    strength_threshold::F
    level_strength_thresholds::L
    smoother_weight::F
    near_nullspace::C
    max_prolongation_entries::I
    aggressive_levels::I
    aggressive_passes::I
    coarse_drop_tolerances::D
    interpolation::S
end

function SmoothAggregation(;
    strength_threshold=0.10,
    level_strength_thresholds=(0.10, 0.075, 0.05),
    smoother_weight=1.0,
    near_nullspace=nothing,
    max_prolongation_entries=2,
    aggressive_levels=1,
    aggressive_passes=1,
    coarse_drop_tolerances=(0.0, 0.01, 0.03, 0.05),
    interpolation=:smoothed
)
    aggressive_levels >= 0 || throw(ArgumentError("SmoothAggregation aggressive_levels must be nonnegative"))
    aggressive_passes > 0 || throw(ArgumentError("SmoothAggregation aggressive_passes must be positive"))
    interpolation in (:smoothed, :unsmoothed, :direct) ||
        throw(ArgumentError("SmoothAggregation interpolation must be :smoothed, :unsmoothed, or :direct"))
    thresholds = isnothing(level_strength_thresholds) ? nothing : float.(collect(level_strength_thresholds))
    drop_tolerances = float.(collect(coarse_drop_tolerances))
    any(<(0), drop_tolerances) && throw(ArgumentError("SmoothAggregation coarse_drop_tolerances must be nonnegative"))
    return SmoothAggregation{
        typeof(float(strength_threshold)),
        typeof(near_nullspace),
        typeof(thresholds),
        Int,
        typeof(drop_tolerances),
        typeof(interpolation)
    }(
        float(strength_threshold),
        thresholds,
        float(smoother_weight),
        near_nullspace,
        Int(max_prolongation_entries),
        Int(aggressive_levels),
        Int(aggressive_passes),
        drop_tolerances,
        interpolation
    )
end

Adapt.@adapt_structure SmoothAggregation

struct RugeStuben{F} <: AbstractAMGCoarsening
    strength_threshold::F
end

RugeStuben(; strength_threshold=0.05) = RugeStuben(float(strength_threshold))
Adapt.@adapt_structure RugeStuben

struct AMGJacobi{F} <: AbstractAMGSmoother
    omega::F
end

AMGJacobi(; omega=0.6667) = AMGJacobi(float(omega))
Adapt.@adapt_structure AMGJacobi

struct AMGChebyshev{F,I} <: AbstractAMGSmoother
    degree::I
    eig_ratio::F
    lambda_scale::F
end

function AMGChebyshev(; degree=2, eig_ratio=30.0, lambda_scale=1.1)
    degree > 0 || throw(ArgumentError("AMGChebyshev degree must be positive"))
    eig_ratio > 1 || throw(ArgumentError("AMGChebyshev eig_ratio must be greater than one"))
    lambda_scale > 0 || throw(ArgumentError("AMGChebyshev lambda_scale must be positive"))
    return AMGChebyshev(Int(degree), float(eig_ratio), float(lambda_scale))
end

Adapt.@adapt_structure AMGChebyshev

struct AMG{M,C,S,Y,I} <: AbstractLinearSolver
    mode::M
    coarsening::C
    smoother::S
    cycle::Y
    pre_sweeps::I
    post_sweeps::I
    max_levels::I
    min_coarse_rows::I
    max_coarse_rows::I
end

_amg_mode(mode::AMGSolver) = mode
_amg_mode(mode::Cg) = mode
_amg_mode(mode) = throw(ArgumentError("AMG mode must be AMGSolver() or Cg()"))

_amg_mode_name(::AMGSolver) = "solve"
_amg_mode_name(::Cg) = "cg"
_amg_mode_name(mode) = string(nameof(typeof(mode)))

_amg_cycle(cycle::VCycle) = cycle
_amg_cycle(cycle) = throw(ArgumentError("AMG only supports cycle=VCycle()"))

_amg_coarsening(coarsening::Union{SmoothAggregation,RugeStuben}) = coarsening
_amg_coarsening(coarsening) =
    throw(ArgumentError("AMG coarsening must be SmoothAggregation(...) or RugeStuben(...)"))

_amg_smoother(smoother::Union{AMGJacobi,AMGChebyshev}) = smoother
_amg_smoother(smoother) =
    throw(ArgumentError("AMG smoother must be AMGJacobi(...) or AMGChebyshev(...)"))

function AMG(;
    mode=Cg(),
    coarsening=SmoothAggregation(),
    smoother=AMGJacobi(),
    cycle=VCycle(),
    pre_sweeps=nothing,
    post_sweeps=nothing,
    max_levels=10,
    min_coarse_rows=32,
    max_coarse_rows=512
)
    mode = _amg_mode(mode)
    coarsening = _amg_coarsening(coarsening)
    smoother = _amg_smoother(smoother)
    cycle = _amg_cycle(cycle)
    default_sweeps = smoother isa AMGChebyshev ? 1 : (mode isa Cg ? 2 : 1)
    pre_sweeps = isnothing(pre_sweeps) ? default_sweeps : pre_sweeps
    post_sweeps = isnothing(post_sweeps) ? default_sweeps : post_sweeps
    pre_sweeps > 0 || throw(ArgumentError("AMG pre_sweeps must be positive"))
    post_sweeps > 0 || throw(ArgumentError("AMG post_sweeps must be positive"))
    return AMG(
        mode,
        coarsening,
        smoother,
        cycle,
        Int(pre_sweeps),
        Int(post_sweeps),
        Int(max_levels),
        Int(min_coarse_rows),
        Int(max_coarse_rows)
    )
end

Adapt.@adapt_structure AMG

mutable struct AMGTimingStats{F,I,S}
    build_time_s::F
    build_calls::I
    refresh_time_s::F
    refresh_calls::I
    finest_refresh_time_s::F
    finest_refresh_calls::I
    apply_time_s::F
    apply_calls::I
    last_update_action::S
end

AMGTimingStats() = AMGTimingStats(0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, :none)

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

mutable struct AMGLevel{MA,MP,MR,VD,VI,VX,T}
    A::MA
    P::MP
    R::MR
    diagonal::VD
    inv_diagonal::VD
    rhs::VX
    x::VX
    tmp::VX
    aggregate_ids::VI
    lambda_max::T
    level_id::Int
    has_transfer::Bool
end

Adapt.@adapt_structure AMGLevel

mutable struct AMGCPUCoarseLevel{MA,MC,VX,LUF,QRF}
    A::MA
    Acsc::MC
    rhs::VX
    x::VX
    lu_factor::LUF
    qr_factor::QRF
    use_qr::Bool
end

mutable struct AMGHierarchy{LD,LH,CC,B,RP,CP}
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
    is_symmetric::Bool
    operator_complexity::Float64
    grid_complexity::Float64
    last_cycle_factor::Float64
end

mutable struct AMGWorkspace{H,TS,V,T,RH} <: AbstractAMGWorkspace
    hierarchy::H
    timing::TS
    solution::V
    residual::V
    correction::V
    search::V
    preconditioned::V
    q::V
    iterations::Int
    last_relative_residual::T
    residual_history::RH
end

function _amg_backend_array(backend, values)
    return adapt(backend, values)
end

function _empty_amg_matrix(backend, ::Type{T}) where {T}
    rowptr = KernelAbstractions.zeros(backend, Int, 1)
    colval = KernelAbstractions.zeros(backend, Int, 0)
    nzval = KernelAbstractions.zeros(backend, T, 0)
    return AMGMatrixCSR(rowptr, colval, nzval, 0, 0)
end

function _empty_amg_level(backend, ::Type{T}) where {T}
    A = _empty_amg_matrix(backend, T)
    P = _empty_amg_matrix(backend, T)
    R = _empty_amg_matrix(backend, T)
    diag = KernelAbstractions.zeros(backend, T, 0)
    invdiag = KernelAbstractions.zeros(backend, T, 0)
    rhs = KernelAbstractions.zeros(backend, T, 0)
    x = KernelAbstractions.zeros(backend, T, 0)
    tmp = KernelAbstractions.zeros(backend, T, 0)
    aggregate_ids = KernelAbstractions.zeros(backend, Int, 0)
    return AMGLevel(A, P, R, diag, invdiag, rhs, x, tmp, aggregate_ids, zero(T), 0, false)
end

function _placeholder_lu_qr(::Type{T}) where {T}
    A = sparse([1], [1], [one(T)], 1, 1)
    return lu(A), qr(A)
end

function _empty_cpu_coarse_level(::Type{T}) where {T}
    lu_factor, qr_factor = _placeholder_lu_qr(T)
    A = AMGMatrixCSR([1, 1], Int[], T[], 1, 1)
    Acsc = sparse([1], [1], [one(T)], 1, 1)
    rhs = zeros(T, 1)
    x = zeros(T, 1)
    return AMGCPUCoarseLevel(A, Acsc, rhs, x, lu_factor, qr_factor, false)
end

function _empty_hierarchy(backend, ::Type{T}) where {T}
    host_level = _empty_amg_level(CPU(), T)
    device_level = _empty_amg_level(backend, T)
    host_levels = typeof(host_level)[]
    device_levels = typeof(device_level)[]
    coarse_cpu = _empty_cpu_coarse_level(T)
    return AMGHierarchy(
        device_levels,
        host_levels,
        coarse_cpu,
        backend,
        1,
        0,
        0,
        Int[],
        Int[],
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        UInt64(0),
        Any[],
        true,
        1.0,
        1.0,
        0.0
    )
end

function _workspace(::AMG, b)
    T = eltype(b)
    backend = KernelAbstractions.get_backend(b)
    x = similar(b)
    return AMGWorkspace(
        _empty_hierarchy(backend, T),
        AMGTimingStats(),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        0,
        zero(T),
        Float64[]
    )
end
