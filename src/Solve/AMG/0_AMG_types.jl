export AMG, AMGSolver, VCycle, SmoothAggregation, RugeStuben, Geometric, AMGJacobi, AMGChebyshev
export AMGGaussSeidel, AMGSOR, AMGSymmetricSweep, AMGForwardSweep, AMGBackwardSweep
export AbstractAMGCPUSmoother, AbstractAMGGPUSmoother
export AMGWorkspace, AMGHierarchy, AMGLevel, AMGTimingStats, AMGRAPPlanCPU

abstract type AbstractAMGMode end
abstract type AbstractAMGCoarsening end
abstract type AbstractAMGSmoother end
abstract type AbstractAMGCPUSmoother <: AbstractAMGSmoother end
abstract type AbstractAMGGPUSmoother <: AbstractAMGSmoother end
abstract type AbstractAMGSweep end
abstract type AbstractAMGCycle end
abstract type AbstractAMGWorkspace end

struct AMGSolver <: AbstractAMGMode end
struct VCycle <: AbstractAMGCycle end
struct AMGSymmetricSweep <: AbstractAMGSweep end
struct AMGForwardSweep <: AbstractAMGSweep end
struct AMGBackwardSweep <: AbstractAMGSweep end

Adapt.@adapt_structure AMGSymmetricSweep
Adapt.@adapt_structure AMGForwardSweep
Adapt.@adapt_structure AMGBackwardSweep

struct SmoothAggregation{F,C} <: AbstractAMGCoarsening
    strength_threshold::F
    smoother_weight::F
    near_nullspace::C
end

function SmoothAggregation(;
    strength_threshold=0.0,
    smoother_weight=4/3,
    near_nullspace=nothing,
)
    return SmoothAggregation{typeof(float(strength_threshold)), typeof(near_nullspace)}(
        float(strength_threshold),
        float(smoother_weight),
        near_nullspace
    )
end

Adapt.@adapt_structure SmoothAggregation

struct RugeStuben{F} <: AbstractAMGCoarsening
    strength_threshold::F
end

RugeStuben(; strength_threshold=0.25) = RugeStuben(float(strength_threshold))
Adapt.@adapt_structure RugeStuben

"""
    Geometric(; merge_levels=1)

OpenFOAM-style geometric agglomeration (`faceAreaPair`): cells are merged pairwise into
coarse clusters by greatest face weight, using `|a_ij|` as the face weight (proportional to
face-area/delta in FVM). `merge_levels` sets the number of pairwise passes per coarse level,
giving cluster size ~`2^merge_levels`. Prolongation is unsmoothed piecewise-constant injection
(additive correction); coarse operators are formed algebraically by Galerkin RAP.
"""
struct Geometric{I} <: AbstractAMGCoarsening
    merge_levels::I
end

function Geometric(; merge_levels=1)
    merge_levels > 0 || throw(ArgumentError("Geometric merge_levels must be positive"))
    return Geometric(Int(merge_levels))
end

Adapt.@adapt_structure Geometric

struct AMGJacobi{F} <: AbstractAMGGPUSmoother
    omega::F
end

# Default omega=4/3: optimal weight for lambda_max-scaled Jacobi targeting upper-half
# spectrum [lmax/2, lmax]; omega_eff = (4/3)/lambda_max. Evidence: see amg_findings.md.
AMGJacobi(; omega=4/3) = AMGJacobi(float(omega))
Adapt.@adapt_structure AMGJacobi

struct AMGChebyshev{F,I} <: AbstractAMGGPUSmoother
    degree::I
    eig_ratio::F
    lambda_scale::F
end

function AMGChebyshev(; degree=3, eig_ratio=10.0, lambda_scale=1.1)
    degree > 0 || throw(ArgumentError("AMGChebyshev degree must be positive"))
    eig_ratio > 1 || throw(ArgumentError("AMGChebyshev eig_ratio must be greater than one"))
    lambda_scale > 0 || throw(ArgumentError("AMGChebyshev lambda_scale must be positive"))
    return AMGChebyshev(Int(degree), float(eig_ratio), float(lambda_scale))
end

Adapt.@adapt_structure AMGChebyshev

struct AMGGaussSeidel{SW,I} <: AbstractAMGCPUSmoother
    sweep::SW
    iterations::I
end

function AMGGaussSeidel(; sweep=AMGSymmetricSweep(), iterations=1)
    sweep isa AbstractAMGSweep || throw(ArgumentError("AMGGaussSeidel sweep must be AMGSymmetricSweep(), AMGForwardSweep(), or AMGBackwardSweep()"))
    iterations > 0 || throw(ArgumentError("AMGGaussSeidel iterations must be positive"))
    return AMGGaussSeidel(sweep, Int(iterations))
end

AMGGaussSeidel(sweep::AbstractAMGSweep; iterations=1) = AMGGaussSeidel(; sweep, iterations)
Adapt.@adapt_structure AMGGaussSeidel

struct AMGSOR{SW,F,I} <: AbstractAMGCPUSmoother
    sweep::SW
    omega::F
    iterations::I
end

function AMGSOR(; omega=1.0, sweep=AMGSymmetricSweep(), iterations=1)
    sweep isa AbstractAMGSweep || throw(ArgumentError("AMGSOR sweep must be AMGSymmetricSweep(), AMGForwardSweep(), or AMGBackwardSweep()"))
    omega > 0 || throw(ArgumentError("AMGSOR omega must be positive"))
    iterations > 0 || throw(ArgumentError("AMGSOR iterations must be positive"))
    return AMGSOR(sweep, float(omega), Int(iterations))
end

AMGSOR(omega; sweep=AMGSymmetricSweep(), iterations=1) = AMGSOR(; omega, sweep, iterations)
AMGSOR(omega, sweep::AbstractAMGSweep; iterations=1) = AMGSOR(; omega, sweep, iterations)
Adapt.@adapt_structure AMGSOR

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
    coarse_refresh_interval::I
end

_amg_mode(mode::AMGSolver) = mode
_amg_mode(mode::Cg) = mode
_amg_mode(mode) = throw(ArgumentError("AMG mode must be AMGSolver() or Cg()"))

_amg_mode_name(::AMGSolver) = "solve"
_amg_mode_name(::Cg) = "cg"
_amg_mode_name(mode) = string(nameof(typeof(mode)))

_amg_cycle(cycle::VCycle) = cycle
_amg_cycle(cycle) = throw(ArgumentError("AMG only supports cycle=VCycle()"))

_amg_coarsening(coarsening::Union{SmoothAggregation,RugeStuben,Geometric}) = coarsening
_amg_coarsening(coarsening) =
    throw(ArgumentError("AMG coarsening must be SmoothAggregation(...), RugeStuben(...), or Geometric(...)"))

_amg_smoother(smoother::Union{AMGJacobi,AMGChebyshev,AMGGaussSeidel,AMGSOR}) = smoother
_amg_smoother(smoother) =
    throw(ArgumentError("AMG smoother must be AMGJacobi(...), AMGChebyshev(...), AMGGaussSeidel(...), or AMGSOR(...)"))

_amg_smoother_name(T) = string(nameof(T))
_amg_supported_smoother_types() = (AMGJacobi, AMGChebyshev, AMGGaussSeidel, AMGSOR)

function _amg_gpu_smoother_names()
    names = sort(_amg_smoother_name.(T for T in _amg_supported_smoother_types() if T <: AbstractAMGGPUSmoother))
    return join(names, ", ")
end

_validate_amg_smoother_backend(::CPU, smoother::AbstractAMGSmoother) = smoother
_validate_amg_smoother_backend(::CPU, smoother::AbstractAMGGPUSmoother) = smoother
_validate_amg_smoother_backend(::CPU, smoother::AbstractAMGCPUSmoother) = smoother
_validate_amg_smoother_backend(backend, smoother::AbstractAMGGPUSmoother) = smoother

function _validate_amg_smoother_backend(backend, smoother::AbstractAMGCPUSmoother)
    throw(ArgumentError(
        "$(nameof(typeof(smoother))) is a sequential CPU AMG smoother and cannot be used with $(nameof(typeof(backend)))(). " *
        "GPU-capable AMG smoothers are: $(_amg_gpu_smoother_names())."
    ))
end

function AMG(;
    mode=Cg(),
    coarsening=SmoothAggregation(),
    smoother=AMGJacobi(),
    cycle=VCycle(),
    pre_sweeps=nothing,
    post_sweeps=nothing,
    max_levels=10,
    min_coarse_rows=32,
    max_coarse_rows=4096,
    coarse_refresh_interval=1
)
    mode = _amg_mode(mode)
    coarsening = _amg_coarsening(coarsening)
    smoother = _amg_smoother(smoother)
    cycle = _amg_cycle(cycle)
    default_sweeps = smoother isa Union{AMGChebyshev,AMGGaussSeidel,AMGSOR} ? 1 : (mode isa Cg ? 2 : 1)
    pre_sweeps = isnothing(pre_sweeps) ? default_sweeps : pre_sweeps
    post_sweeps = isnothing(post_sweeps) ? default_sweeps : post_sweeps
    pre_sweeps > 0 || throw(ArgumentError("AMG pre_sweeps must be positive"))
    post_sweeps > 0 || throw(ArgumentError("AMG post_sweeps must be positive"))
    coarse_refresh_interval > 0 || throw(ArgumentError("AMG coarse_refresh_interval must be positive"))
    return AMG(
        mode,
        coarsening,
        smoother,
        cycle,
        Int(pre_sweeps),
        Int(post_sweeps),
        Int(max_levels),
        Int(min_coarse_rows),
        Int(max_coarse_rows),
        Int(coarse_refresh_interval)
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

struct AMGGalerkinCache{I,W}
    targets::Vector{I}
    fine_indices::Vector{I}
    weights::Vector{W}
end

# Symbolic/numeric split RAP plan (PETSc/Ginkgo style).
# Stores the intermediate R×A sparsity pattern and a pre-allocated value buffer.
# Refresh is pattern-free: fill ra_nzval, then scatter into coarse_A.nzval — no allocation.
# Backend extensions replace this with device-resident plans (e.g. AMGRAPPlanCUDA).
struct AMGRAPPlanCPU{I, T}
    ra_rowptr::Vector{Int}  # R×A row pointers
    ra_colval::Vector{I}    # R×A column indices (sorted per row)
    ra_nzval::Vector{T}     # R×A values (mutable buffer)
    # SPA (Sparse Pointer Array) workspaces for O(1) scatter — avoids binary search on refresh
    workspace_ra::Vector{T}  # dense, size = n_fine
    workspace_rap::Vector{T} # dense, size = n_coarse_out
    flag_ra::Vector{Int}     # stamp array; flag_ra[j]==r means workspace_ra[j] set for row r
    flag_rap::Vector{Int}    # same for RAP accumulation
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
    coarse_tmp::VX
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
    galerkin_caches::Vector{Any}
    is_symmetric::Bool
    coarse_rhs_copy_time_s::Float64
    coarse_cpu_solve_time_s::Float64
    coarse_x_copy_time_s::Float64
    coarse_solve_calls::Int
    coarse_device_solve_time_s::Float64
    coarse_device_solve_calls::Int
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
    diag_index = KernelAbstractions.zeros(backend, Int, 0)
    rhs = KernelAbstractions.zeros(backend, T, 0)
    x = KernelAbstractions.zeros(backend, T, 0)
    tmp = KernelAbstractions.zeros(backend, T, 0)
    direction = KernelAbstractions.zeros(backend, T, 0)
    coarse_tmp = KernelAbstractions.zeros(backend, T, 0)
    aggregate_ids = KernelAbstractions.zeros(backend, Int, 0)
    return AMGLevel(A, P, R, diag, invdiag, diag_index, rhs, x, tmp, direction, coarse_tmp, aggregate_ids, zero(T), 0, false)
end

function _placeholder_lu_qr(::Type{T}) where {T}
    A = sparse([1], [1], [one(T)], 1, 1)
    return lu(A), qr(A)
end

function _empty_cpu_coarse_level(::Type{T}) where {T}
    lu_factor, qr_factor = _placeholder_lu_qr(T)
    A = AMGMatrixCSR([1, 1], Int[], T[], 1, 1)
    Acsc = sparse([1], [1], [one(T)], 1, 1)
    csc_nzval_index = Int[]
    rhs = zeros(T, 1)
    x = zeros(T, 1)
    return AMGCPUCoarseLevel(A, Acsc, csc_nzval_index, rhs, x, lu_factor, qr_factor, false)
end

function _empty_hierarchy(backend, ::Type{T}) where {T}
    host_level = _empty_amg_level(CPU(), T)
    device_level = _empty_amg_level(backend, T)
    host_levels = typeof(host_level)[]
    # GPU levels are heterogeneously typed (finest may be CuSparseMatrixCSR); use Any on device backends
    device_levels = backend isa CPU ? typeof(device_level)[] : Vector{Any}()
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
        Any[],
        true,
        0.0,
        0.0,
        0.0,
        0,
        0.0,
        0,
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
