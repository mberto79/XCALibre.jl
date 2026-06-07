export AMG, AMGSolver, VCycle, SmoothAggregation, RugeStuben, Geometric, AMGJacobi, AMGChebyshev
export AMGGaussSeidel, AMGSOR, AMGSymmetricSweep, AMGForwardSweep, AMGBackwardSweep
export AbstractAMGCPUSmoother, AbstractAMGGPUSmoother, OnDevice, OnHost, OnDeviceKrylov
export OnDeviceJacobi, OnDeviceChebyshev
export AMGWorkspace, AMGHierarchy, AMGLevel, AMGRAPPlanCPU

abstract type AbstractAMGMode end
abstract type AbstractAMGCoarsening end
abstract type AbstractAMGSmoother end
abstract type AbstractAMGCPUSmoother <: AbstractAMGSmoother end
abstract type AbstractAMGGPUSmoother <: AbstractAMGSmoother end
abstract type AbstractAMGSweep end
abstract type AbstractAMGCycle end
abstract type AbstractAMGCoarseSolve end
abstract type AbstractAMGWorkspace end

struct AMGSolver <: AbstractAMGMode end
struct VCycle <: AbstractAMGCycle end

"""
    OnHost()

Solve the coarsest level on the host (sparse LU/QR). On a GPU backend the per-cycle coarse solve
copies the rhs to host and the solution back — fastest single-device for a largish coarsest.
"""
struct OnHost <: AbstractAMGCoarseSolve end

"""
    OnDevice(; max_rows=512)

Solve the coarsest level on the device (no host round-trip): a device-resident factorization
(Cholesky/LU on CUDA; generic backends use a device dense inverse + GEMV fallback). Eliminates a
per-cycle fine→coarse host sync/transfer (the win at MPI/multi-GPU scale). The factor is rebuilt
each coarse refresh (~n^3), so coarsest levels larger than `max_rows` fall back to `OnHost` —
`max_rows=512` is a conservative no-regression default (well-coarsened 3D reaches a tiny coarsest).
On a CPU backend this is equivalent to `OnHost`.
"""
struct OnDevice{I} <: AbstractAMGCoarseSolve
    max_rows::I
end

OnDevice(; max_rows=512) = OnDevice(Int(max_rows))
Adapt.@adapt_structure OnDevice
Adapt.@adapt_structure OnHost

"""
    OnDeviceKrylov(; solver=nothing, rtol=1e-2, atol=0.0, itmax=50)

GPU strategy: truncate the hierarchy at a LARGE coarsest level (set `max_coarse_rows` high, e.g.
4000-8000) so every level has enough work for good GPU occupancy, then solve that coarsest level
with an in-place Krylov solver (Cg/Bicgstab) + Jacobi preconditioner — the configuration GPUs
parallelise best. Reuses XCALibre's Krylov.jl infrastructure (no host round-trip, no extra alloc).
`solver=nothing` auto-picks Cg (symmetric coarsest) or Bicgstab. GATED to `mode=AMGSolver()`: an
inner Krylov solve is nonlinear in b, which breaks PCG's fixed-SPD-preconditioner assumption. On a
CPU backend this is equivalent to `OnHost`.
"""
struct OnDeviceKrylov{S,F,I} <: AbstractAMGCoarseSolve
    solver::S
    rtol::F
    atol::F
    itmax::I
end

OnDeviceKrylov(; solver=nothing, rtol=1e-2, atol=0.0, itmax=50) =
    OnDeviceKrylov(_amg_krylov_solver(solver), float(rtol), float(atol), Int(itmax))
Adapt.@adapt_structure OnDeviceKrylov

_amg_krylov_solver(::Nothing) = nothing
_amg_krylov_solver(s::Union{Cg,Bicgstab}) = s
_amg_krylov_solver(s) =
    throw(ArgumentError("OnDeviceKrylov solver must be Cg(), Bicgstab(), or nothing (auto)"))

"""
    OnDeviceJacobi(; omega=4/3, iterations=10)

GPU coarse solve: apply a FIXED number of weighted-Jacobi sweeps (x init 0) to the truncated,
large coarsest level instead of a direct factor. A fixed-sweep Jacobi is a constant linear operator
`p(A_c)·b`, so — unlike [`OnDeviceKrylov`](@ref) — it is valid as an outer-CG preconditioner: allowed
in BOTH `mode=Cg()` and `mode=AMGSolver()`. Fully on device (no host round-trip), removing the coarse
host sync point at MPI/multi-GPU scale. Cheaper per cycle than a direct solve but only approximate, so
outer iterations may rise; best when the coarsest is large and well-conditioned (huge coarsening
ratio). On a CPU backend this is equivalent to `OnHost`.
"""
struct OnDeviceJacobi{F,I} <: AbstractAMGCoarseSolve
    omega::F
    iterations::I
end

function OnDeviceJacobi(; omega=4/3, iterations=10)
    iterations > 0 || throw(ArgumentError("OnDeviceJacobi iterations must be positive"))
    return OnDeviceJacobi(float(omega), Int(iterations))
end
Adapt.@adapt_structure OnDeviceJacobi

"""
    OnDeviceChebyshev(; degree=10, eig_ratio=10.0, lambda_scale=1.1)

GPU coarse solve: apply a FIXED-degree Chebyshev polynomial smoother (x init 0) to the coarsest
level. Like [`OnDeviceJacobi`](@ref) it is a constant linear operator (b-independent), so valid in
BOTH `mode=Cg()` and `mode=AMGSolver()`; fully on device. Chebyshev targets the spectrum
`[lambda_max/eig_ratio, lambda_max]` (lambda_max estimated per level; `lambda_scale` inflates the
upper bound), giving faster coarse error reduction per matvec than Jacobi at equal cost. On a CPU
backend this is equivalent to `OnHost`.
"""
struct OnDeviceChebyshev{I,F} <: AbstractAMGCoarseSolve
    degree::I
    eig_ratio::F
    lambda_scale::F
end

function OnDeviceChebyshev(; degree=10, eig_ratio=10.0, lambda_scale=1.1)
    degree > 0 || throw(ArgumentError("OnDeviceChebyshev degree must be positive"))
    eig_ratio > 1 || throw(ArgumentError("OnDeviceChebyshev eig_ratio must be greater than one"))
    lambda_scale > 0 || throw(ArgumentError("OnDeviceChebyshev lambda_scale must be positive"))
    return OnDeviceChebyshev(Int(degree), float(eig_ratio), float(lambda_scale))
end
Adapt.@adapt_structure OnDeviceChebyshev

_amg_coarse_solve(cs::AbstractAMGCoarseSolve) = cs
_amg_coarse_solve(::CPU) = OnHost()  # convenience: coarse_solve=CPU() means host solve
_amg_coarse_solve(::KernelAbstractions.GPU) = OnDeviceKrylov()  # coarse_solve=CUDABackend()
_amg_coarse_solve(cs) =
    throw(ArgumentError("AMG coarse_solve must be OnDevice(...), OnDeviceJacobi(...), OnDeviceChebyshev(...), OnDeviceKrylov(...), OnHost(), CPU(), or a GPU backend"))
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

"""
    AMGJacobi(; omega=4/3)

Weighted-Jacobi AMG smoother. `omega` is a **lambda_max-scaled** coefficient: the effective
relaxation factor applied each sweep is `ω_eff = omega / lambda_max`, where `lambda_max` is
the spectral radius of `D⁻¹A` at that level. The default `omega=4/3` gives `ω_eff ≈ 2/3`
for Poisson-like operators (where `lambda_max ≈ 2`), which damps the upper half of the
spectrum optimally. Valid scaled range: `(0, 2)` — values ≥ 2 are clamped for SPD stability.
Note: unlike [`AMGSOR`](@ref), `omega` here is NOT a direct relaxation factor.
"""
struct AMGJacobi{F} <: AbstractAMGGPUSmoother
    omega::F
end

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

"""
    AMGGaussSeidel(; sweep=AMGSymmetricSweep(), iterations=1)

Gauss-Seidel AMG smoother (CPU only). `sweep` selects the traversal order:
`AMGSymmetricSweep()` (forward + backward), `AMGForwardSweep()`, or `AMGBackwardSweep()`.
`iterations` sets the number of sweeps per smoother call. `omega` is fixed at 1 (no
relaxation); use [`AMGSOR`](@ref) for a tunable direct relaxation factor.
"""
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

"""
    AMGSOR(; omega=1.0, sweep=AMGSymmetricSweep(), iterations=1)

Successive Over-Relaxation AMG smoother (CPU only). `omega` is the **direct** relaxation
factor applied each sweep: `x_new = (1-omega)*x_old + omega*(D\\(b - L*x))`. Stable range
`(0, 2)`; `omega=1` recovers Gauss-Seidel. Unlike [`AMGJacobi`](@ref), `omega` is NOT
scaled by `lambda_max` — it is used as-is.
"""
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

struct AMG{M,C,S,Y,CS,I} <: AbstractLinearSolver
    mode::M
    coarsening::C
    smoother::S
    cycle::Y
    coarse_solve::CS
    scale_correction::Bool
    pre_sweeps::I
    post_sweeps::I
    max_levels::I
    min_coarse_rows::I
    max_coarse_rows::I
    coarse_refresh_interval::I
    fuse_levels::I
    coarse_storage::DataType
end

# Storage precision of the multigrid hierarchy. Float32 stores every level's operators/transfers/
# work vectors (incl. finest cycle copy) in single precision — halves SpMV/smoother/RAP bandwidth —
# while the outer Krylov/defect-correction stays Float64 (system matrix + solution). Both modes.
_amg_storage(::Type{Float64}) = Float64
_amg_storage(::Type{Float32}) = Float32
_amg_storage(s) = throw(ArgumentError("AMG coarse_storage must be Float32 or Float64"))

# Effective cycle storage = min precision of (working T, requested TS). Storing the hierarchy at
# HIGHER precision than the working/system matrix wastes memory with no accuracy gain (e.g. Float32
# mesh + default coarse_storage=Float64 would otherwise upcast the whole hierarchy). Clamping keeps
# a Float32 mesh in Float32 regardless of coarse_storage; Float64 mesh + Float32 storage = mixed.
_effective_storage(::Type{T}, ::Type{TS}) where {T,TS} = sizeof(TS) >= sizeof(T) ? T : TS

# SuiteSparse (CHOLMOD/UMFPACK/SPQR) sparse factorizations support only Float64/ComplexF64, so the
# host coarsest DIRECT solve always runs in Float64. The coarsest level is tiny (<= max_coarse_rows),
# so this is cheap and numerically robust; rhs/solution are converted at the precision boundary.
_coarse_direct_eltype(::Type{Float32}) = Float64
_coarse_direct_eltype(::Type{T}) where {T} = T

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
    coarse_solve=OnDevice(),
    scale_correction=true,
    pre_sweeps=nothing,
    post_sweeps=nothing,
    max_levels=10,
    min_coarse_rows=32,
    max_coarse_rows=4096,
    coarse_refresh_interval=1,
    fuse_levels=0,
    coarse_storage=Float64
)
    mode = _amg_mode(mode)
    coarsening = _amg_coarsening(coarsening)
    smoother = _amg_smoother(smoother)
    cycle = _amg_cycle(cycle)
    coarse_solve = _amg_coarse_solve(coarse_solve)
    if coarse_solve isa OnDeviceKrylov && mode isa Cg
        throw(ArgumentError("coarse_solve=OnDeviceKrylov requires mode=AMGSolver() (an inner Krylov coarse solve is nonlinear and breaks AMG-preconditioned CG)"))
    end
    # Jacobi defaults to V(2,2): the standalone AMGSolver V-cycle has no Krylov acceleration, so
    # extra cheap smoothing cuts cycle count (fewer coarse-grid sync points); on GPU per-cycle
    # launch/sync overhead makes V(2,2) clearly faster than V(1,1). Cg unchanged (was already 2).
    default_sweeps = smoother isa Union{AMGChebyshev,AMGGaussSeidel,AMGSOR} ? 1 : 2
    pre_sweeps = isnothing(pre_sweeps) ? default_sweeps : pre_sweeps
    post_sweeps = isnothing(post_sweeps) ? default_sweeps : post_sweeps
    pre_sweeps > 0 || throw(ArgumentError("AMG pre_sweeps must be positive"))
    post_sweeps > 0 || throw(ArgumentError("AMG post_sweeps must be positive"))
    coarse_refresh_interval > 0 || throw(ArgumentError("AMG coarse_refresh_interval must be positive"))
    # fuse_levels: opt-in to the matrix-free greenfield GPU path (>0 on a GPU backend with Geometric
    # +AMGJacobi). Default 0 = reference materialised V-cycle. The path is implemented but experimental
    # (saves VRAM, see device/); flip on per-solver, not by default. Negative is invalid.
    fuse_levels >= 0 || throw(ArgumentError("AMG fuse_levels must be non-negative (0 disables the fused GPU path)"))
    return AMG(
        mode,
        coarsening,
        smoother,
        cycle,
        coarse_solve,
        Bool(scale_correction),
        Int(pre_sweeps),
        Int(post_sweeps),
        Int(max_levels),
        Int(min_coarse_rows),
        Int(max_coarse_rows),
        Int(coarse_refresh_interval),
        Int(fuse_levels),
        _amg_storage(coarse_storage)
    )
end

Adapt.@adapt_structure AMG

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
    operator_complexity::Float64
    grid_complexity::Float64
    last_cycle_factor::Float64
    coarse_inv::Base.RefValue{Any}  # device dense inverse of coarsest A (GPU); nothing on CPU/oversized
    cycle_input::Base.RefValue{Any}  # mixed precision: TS-typed finest rhs buffer; nothing when TS==TW
    greenfield::Base.RefValue{Any}  # greenfield GPU pipeline state (MFGreenfield); nothing on the reference path
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
    iterations::Int
    converged::Bool
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
    # A holds the working-precision (T) coarsest CSR for pattern caching / nzval sync; the direct
    # factor (Acsc/lu/qr/rhs/x) runs at TC=Float64 because SuiteSparse rejects Float32 (see above).
    TC = _coarse_direct_eltype(T)
    lu_factor, qr_factor = _placeholder_lu_qr(TC)
    A = AMGMatrixCSR([1, 1], Int[], T[], 1, 1)
    Acsc = sparse([1], [1], [one(TC)], 1, 1)
    csc_nzval_index = Int[]
    rhs = zeros(TC, 1)
    x = zeros(TC, 1)
    return AMGCPUCoarseLevel(A, Acsc, csc_nzval_index, rhs, x, lu_factor, qr_factor, false)
end

function _empty_hierarchy(backend, ::Type{T}, ::Type{TS}=T) where {T,TS}
    host_level = _empty_amg_level(CPU(), T)
    # Mixed precision: device (cycle) levels are stored at TS; the empty must match setup_hierarchy's
    # concrete type so the workspace.hierarchy reassignment after build type-checks.
    device_level = _empty_amg_level(backend, TS)
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
        1.0,
        1.0,
        0.0,
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing)
    )
end

function _workspace(solver::AMG, b)
    T = eltype(b)
    TS = _effective_storage(T, _amg_storage(solver.coarse_storage))
    backend = KernelAbstractions.get_backend(b)
    x = similar(b)
    return AMGWorkspace(
        _empty_hierarchy(backend, T, TS),
        0,
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        0,
        false,
        zero(T),
        Float64[]
    )
end
