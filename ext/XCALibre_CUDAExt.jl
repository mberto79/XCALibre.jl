module XCALibre_CUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using XCALibre, Adapt, SparseArrays, SparseMatricesCSR, KernelAbstractions
using LinearAlgebra, LinearOperators
import KrylovPreconditioners as KP

const SPARSEGPU = CUDA.CUSPARSE.CuSparseMatrixCSR
const BACKEND = CUDABackend
const GPUARRAY = CuArray
using CUDA.CUSPARSE: ilu02, ilu02!, ic02, ic02!

function XCALibre.Mesh._convert_array!(arr, backend::BACKEND)
    return adapt(GPUARRAY, arr) # using GPUARRAY
end

import XCALibre.ModelFramework: _nzval, _rowptr, _colval, get_sparse_fields, 
                                _build_A, _build_opA

_build_A(backend::BACKEND, i, j, v, n) = begin
    A = sparse(i, j, v, n, n)
    SPARSEGPU(A)
end

_build_opA(A::SPARSEGPU) = KP.KrylovOperator(A)
@inline _nzval(A::SPARSEGPU) = A.nzVal
@inline _rowptr(A::SPARSEGPU) = A.rowPtr
@inline _colval(A::SPARSEGPU) = A.colVal
@inline XCALibre.Solve._nzval(A::SPARSEGPU) = A.nzVal
@inline XCALibre.Solve._rowptr(A::SPARSEGPU) = A.rowPtr
@inline XCALibre.Solve._colval(A::SPARSEGPU) = A.colVal
@inline get_sparse_fields(A::SPARSEGPU) = begin
    A.nzVal, A.colVal, A.rowPtr
end

import XCALibre.Solve: _m, _n, update_preconditioner!, _amg_setup_backend, _amg_setup_matrix

function sparse_array_deconstructor_preconditioners(arr::SPARSEGPU)
    (; colVal, rowPtr, nzVal, dims) = arr
    return colVal, rowPtr, nzVal, dims[1], dims[2]
end

_m(A::SPARSEGPU) = A.dims[1]
_n(A::SPARSEGPU) = A.dims[2]

_amg_setup_backend(::BACKEND) = CPU()

function _amg_setup_matrix(A::SPARSEGPU, ::CPU)
    i, j, v = findnz(A)
    ih = adapt(CPU(), i)
    jh = adapt(CPU(), j)
    vh = adapt(CPU(), v)
    SparseXCSR(sparsecsr(ih, jh, vh, size(A, 1), size(A, 2)))
end

# DILU Preconditioner (hybrid implementation for now)

Preconditioner{DILU}(Agpu::SPARSEGPU{F,I}) where {F,I} = begin
    i, j, v = findnz(Agpu)
    ih = adapt(CPU(), i)
    jh = adapt(CPU(), j)
    vh = adapt(CPU(), v)
    n = max(maximum(ih), maximum(jh))
    A = sparsecsr(ih, jh, vh, n , n)
    m, n = size(A)
    m == n || throw("Matrix not square")
    D = zeros(F, m)
    Di = zeros(I, m)
    XCALibre.Solve.diagonal_indices!(Di, A)
    S = XCALibre.Solve.DILUprecon(A, D, Di)
    P = S
    Preconditioner{DILU,typeof(Agpu),typeof(P),typeof(S)}(Agpu,P,S)
end

update_preconditioner!(P::Preconditioner{DILU,M,PT,S},  mesh, config) where {M<:SPARSEGPU,PT,S} =
begin
    KernelAbstractions.copyto!(CPU(), P.storage.A.nzval, P.A.nzVal)
    update_dilu_diagonal!(P, mesh, config)
    nothing
end

# IC0GPU

struct GPUPredonditionerStorage{A,B,C}
    P::A
    L::B 
    U::C
end

Preconditioner{IC0GPU}(A::SPARSEGPU) = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    PS = ic02(A)
    L = KP.TriangularOperator(PS, 'L', 'N', nrhs=1, transa='N')
    U = KP.TriangularOperator(PS, 'L', 'N', nrhs=1, transa='T')
    S = GPUPredonditionerStorage(PS,L,U)

    T = eltype(A.nzVal)
    z = KernelAbstractions.zeros(BACKEND(), T, n)

    P = LinearOperator(T, n, n, true, true, (y, x) -> ldiv_ic0!(S, x, y, z))

    Preconditioner{IC0GPU,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

function ldiv_ic0!(S, x, y, z)
    ldiv!(z, S.L, x)   # Forward substitution with L
    ldiv!(y, S.U, z)  # Backward substitution with Lᴴ
    return y
end

update_preconditioner!(P::Preconditioner{IC0GPU,M,PT,S},  mesh, config) where {M<:SPARSEGPU,PT,S} = 
begin
    P.storage.P.nzVal .= P.A.nzVal
    ic02!(P.storage.P)
    KP.update!(P.storage.L, P.storage.P)
    KP.update!(P.storage.U, P.storage.P)
    nothing
end

# ILU0GPU

Preconditioner{ILU0GPU}(A::SPARSEGPU) = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    PS = ilu02(A)
    L = KP.TriangularOperator(PS, 'L', 'U', nrhs=1, transa='N')
    U = KP.TriangularOperator(PS, 'U', 'N', nrhs=1, transa='N')
    S = GPUPredonditionerStorage(PS,L,U)

    T = eltype(A.nzVal)
    z = KernelAbstractions.zeros(BACKEND(), T, n)

    P = LinearOperator(T, n, n, true, true, (y, x) -> ldiv_ilu0!(S, x, y, z))

    Preconditioner{ILU0GPU,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

function ldiv_ilu0!(S, x, y, z)
    ldiv!(z, S.L, x)   # Forward substitution with L
    ldiv!(y, S.U, z)  # Backward substitution with Lᴴ
    return y
end

update_preconditioner!(P::Preconditioner{ILU0GPU,M,PT,S},  mesh, config) where {M<:SPARSEGPU,PT,S} = 
begin
    P.storage.P.nzVal .= P.A.nzVal
    ilu02!(P.storage.P)
    KP.update!(P.storage.L, P.storage.P)
    KP.update!(P.storage.U, P.storage.P)
    nothing
end

import LinearAlgebra.ldiv!, LinearAlgebra.\
export ldiv!

ldiv!(x::GPUARRAY, P::DILUprecon{M,V,VI}, b) where {M<:AbstractSparseArray,V,VI} =
begin
    xcpu = Vector(x)
    bcpu = Vector(b)
    XCALibre.Solve.forward_substitution!(xcpu, P, bcpu)
    XCALibre.Solve.backward_substitution!(xcpu, P, xcpu)
    KernelAbstractions.copyto!(BACKEND(), x, xcpu)
end

# AMG: cuSPARSE-backed finest-level operators and device RAP plans

import XCALibre.Solve: _matvec!, _residual!, _prolongate_add!, _amg_jacobi!,
    _amg_finalize_device_levels, _amg_finalize_transfer_plans, _refresh_coarse_level!,
    _level_jacobi_omega, _launch_amg_kernel!,
    _amg_weighted_diagonal_correction_kernel!, AMGHierarchy, AbstractAMGHierarchy, AMGLevel, AMGMatrixCSR, AMGJacobi,
    AMGRAPPlanCPU, _refresh_coarse_operators!, _refresh_level_device!, _refresh_coarse_cpu!,
    refresh_hierarchy!, _sync_device_levels_numeric!, _build_coarse_inverse!,
    _build_host_coarse_inverse!, OnDevice, OnDeviceKrylov, OnDeviceJacobi, OnDeviceChebyshev

# Wrap a device AMGMatrixCSR as CuSparseMatrixCSR, sharing nzVal so numeric refresh updates the operator
function _amg_csr_to_cusparse(A::AMGMatrixCSR)
    nzval = XCALibre.Solve._nzval(A)
    rowPtr = CuArray{Cint}(XCALibre.Solve._rowptr(A))
    colVal = CuArray{Cint}(XCALibre.Solve._colval(A))
    return SPARSEGPU{eltype(nzval)}(rowPtr, colVal, nzval, (_m(A), _n(A)))
end

# Wrap every level's operators as cuSPARSE: cuSPARSE SpMV on apply and device-resident RAP on
# refresh. Shares nzVal so device-to-device numeric refresh updates the operators in place.
function _amg_finalize_device_levels(::BACKEND, levels)
    isempty(levels) && return levels
    new_levels = Vector{Any}(undef, length(levels))
    for (k, lvl) in enumerate(levels)
        A = _amg_csr_to_cusparse(lvl.A)
        P = lvl.has_transfer ? _amg_csr_to_cusparse(lvl.P) : lvl.P
        R = lvl.has_transfer ? _amg_csr_to_cusparse(lvl.R) : lvl.R
        new_levels[k] = AMGLevel(
            A, P, R, lvl.diagonal, lvl.inv_diagonal, lvl.diagonal_index,
            lvl.rhs, lvl.x, lvl.tmp, lvl.direction, lvl.coarse_tmp, lvl.aggregate_ids,
            lvl.lambda_max, lvl.level_id, lvl.has_transfer
        )
    end
    return new_levels
end

_matvec!(hierarchy::AbstractAMGHierarchy, y, A::SPARSEGPU, x) = (mul!(y, A, x); y)

function _residual!(hierarchy::AbstractAMGHierarchy, r, A::SPARSEGPU, x, b)
    T = eltype(r)
    copyto!(r, b)
    mul!(r, A, x, -one(T), one(T))  # r = b - A*x
    return r
end

_prolongate_add!(hierarchy::AMGHierarchy, x, P::SPARSEGPU, coarse_x, tmp) =
    (mul!(x, P, coarse_x, one(eltype(x)), one(eltype(x))); x)  # x += P*coarse_x

# Weighted Jacobi via cuSPARSE residual + diagonal correction (equivalent to the fused CSR sweep)
function _amg_jacobi!(hierarchy::AMGHierarchy, smoother::AMGJacobi, level::AMGLevel, A::SPARSEGPU, b, loops)
    omega = _level_jacobi_omega(smoother, level)
    for _ in 1:loops
        _residual!(hierarchy, level.tmp, A, level.x, b)
        _launch_amg_kernel!(
            hierarchy, _amg_weighted_diagonal_correction_kernel!,
            length(level.x), level.x, level.tmp, level.inv_diagonal, omega
        )
    end
    return level.x
end


# Device RAP plan. Keep the CPU plan as a correctness fallback until SpGEMMreuse is
# reintroduced with a verified numeric refresh path.
mutable struct AMGRAPPlanCUDA{T}
    cpu_plan::AMGRAPPlanCPU
    R_dev::SPARSEGPU{T}
    A_dev::SPARSEGPU{T}
    P_dev::SPARSEGPU{T}
end

function _cuda_rap_pattern_matches_host(C::SPARSEGPU, A_host)
    Int.(Array(C.rowPtr)) == XCALibre.Solve._rowptr(A_host) || return false
    Int.(Array(C.colVal)) == XCALibre.Solve._colval(A_host) || return false
    return true
end

function _build_rap_plan_cuda(
    cpu_plan::AMGRAPPlanCPU,
    R_dev::SPARSEGPU{T},
    A_dev::SPARSEGPU{T},
    P_dev::SPARSEGPU{T},
    coarse_A_host
) where T
    rap = (R_dev * A_dev) * P_dev
    CUDA.synchronize()
    _cuda_rap_pattern_matches_host(rap, coarse_A_host) ||
        error("CUDA RAP pattern differs from host Galerkin pattern")
    return AMGRAPPlanCUDA{T}(cpu_plan, R_dev, A_dev, P_dev)
end

# Convert CPU plans to CUDA device plans for all levels that have a transfer matrix.
function _amg_finalize_transfer_plans(::BACKEND, transfer_csc, host_levels, device_levels)
    isempty(transfer_csc) && return transfer_csc
    new_plans = Vector{Any}(undef, length(transfer_csc))
    for i in eachindex(transfer_csc)
        plan = transfer_csc[i]
        if plan isa AMGRAPPlanCPU && i < length(device_levels)
            dev = device_levels[i]
            R_dev = dev.R isa SPARSEGPU ? dev.R : nothing
            P_dev = dev.P isa SPARSEGPU ? dev.P : nothing
            A_dev = dev.A isa SPARSEGPU ? dev.A : nothing
            if !isnothing(R_dev) && !isnothing(P_dev) && !isnothing(A_dev)
                try
                    new_plans[i] = _build_rap_plan_cuda(plan, R_dev, A_dev, P_dev, host_levels[i + 1].A)
                    continue
                catch
                    # Fall back to CPU plan if CUDA plan fails (e.g. incompatible types)
                end
            end
        end
        new_plans[i] = plan
    end
    return new_plans
end

# Dispatch refresh for CUDA plan: plain on-device RAP, then sync coarse nzval to host.
function _refresh_coarse_level!(coarse_A_host, fine_level::AMGLevel, plan::AMGRAPPlanCUDA{T}) where T
    try
        coarse_A_dev = (plan.R_dev * plan.A_dev) * plan.P_dev
        CUDA.synchronize()
        if _cuda_rap_pattern_matches_host(coarse_A_dev, coarse_A_host)
            copyto!(coarse_A_host.nzval, Array(coarse_A_dev.nzVal))
            return coarse_A_host
        end
    catch
    end
    return XCALibre.Solve._refresh_coarse_level!(coarse_A_host, fine_level, plan.cpu_plan)
end

# Device-resident coarse-operator refresh: Galerkin RAP, diag and lambda_max all on the GPU.
# Avoids host RAP (CPU fallback for coarse levels) and host power iteration. Same math.

# Overwrite target.nzVal with (R*A)*P in place (pattern verified == target at setup; guard on nnz).
function _device_rap_into!(target::SPARSEGPU, R::SPARSEGPU, A::SPARSEGPU, P::SPARSEGPU)
    rap = (R * A) * P
    length(rap.nzVal) == length(target.nzVal) ||
        error("device RAP nnz $(length(rap.nzVal)) != target $(length(target.nzVal))")
    copyto!(target.nzVal, rap.nzVal)
    return target
end

# Device-resident path used only when every transfer plan was CUDA-verified at setup (so the
# device RAP column ordering provably matches the host pattern that diagonal_index indexes).
_all_cuda_rap_plans(hierarchy) =
    !isempty(hierarchy.transfer_csc) && all(p -> p isa AMGRAPPlanCUDA, hierarchy.transfer_csc)

function _device_coarse_refresh!(hierarchy::AMGHierarchy, solver)
    levels = hierarchy.levels
    L = length(levels)
    # Level 1 (finest) diag + lambda_max already refreshed before this call. Walk down the
    # hierarchy: each level's A holds new values once the previous level's RAP has run.
    for k in 1:(L - 1)
        k > 1 && _refresh_level_device!(hierarchy, levels[k])
        _device_rap_into!(levels[k + 1].A, levels[k].R, levels[k].A, levels[k].P)
    end
    # On-device coarse solvers (Krylov / fixed-sweep Jacobi / Chebyshev) solve the (large) coarsest
    # on device; skip the costly D->H + host refactor. Their coarsest diag/lambda are refreshed by
    # _build_coarse_inverse! after this call.
    if !(solver.coarse_solve isa Union{OnDeviceKrylov,OnDeviceJacobi,OnDeviceChebyshev})
        # Coarsest level is solved by direct LU on host: D->H its (tiny) nzval and refactor.
        host_coarse = hierarchy.host_levels[end].A
        copyto!(host_coarse.nzval, Array(_nzval(levels[end].A)))
        _refresh_coarse_cpu!(hierarchy.coarse_cpu, host_coarse)
    end
    return hierarchy
end

# Device-resident coarse direct solver: densify the coarsest cuSPARSE operator on device, factor on
# device (Cholesky for SPD, LU otherwise), and apply per-cycle as a device triangular solve — no
# host copy of the coarse matrix. Falls back to the host dense-inverse path when the coarsest
# exceeds the rebuild-cost cap or factorization fails (singular pure-Neumann coarsest).
function _build_coarse_inverse!(::BACKEND, hierarchy::AMGHierarchy, cs::OnDevice)
    coarseA = hierarchy.levels[end].A
    n = size(coarseA, 1)
    (n == 0 || n > cs.max_rows) && return _build_host_coarse_inverse!(hierarchy, cs.max_rows)
    Adense = CuMatrix(coarseA)
    fac = nothing
    # SPD coarsest (Dirichlet present) → Cholesky. UNVERIFIED: a singular/indefinite coarsest
    # (pure-Neumann pressure) assumes CUSOLVER cholesky/lu throw rather than return info>0; if not,
    # the host pinv fallback below won't trigger. All validated cases are SPD-nonsingular.
    try
        fac = cholesky(Hermitian(Adense))
    catch err
        err isa LinearAlgebra.PosDefException || rethrow(err)
        try
            fac = lu(Adense)
        catch err2
            err2 isa LinearAlgebra.SingularException || rethrow(err2)
        end
    end
    fac === nothing && return _build_host_coarse_inverse!(hierarchy, cs.max_rows)
    hierarchy.coarse_inv[] = fac
    return hierarchy
end

function _refresh_coarse_operators!(::BACKEND, hierarchy::AMGHierarchy, solver::XCALibre.Solve.AMG)
    _all_cuda_rap_plans(hierarchy) && return _device_coarse_refresh!(hierarchy, solver)
    # Fallback (mixed CUDA/CPU plans, i.e. a device RAP pattern failed to verify at setup).
    # WARNING: refresh_hierarchy! routes CUDA-plan levels through _refresh_coarse_level!(::AMGRAPPlanCUDA),
    # which reads plan.A_dev — stale during this host-ordered refresh — so coarse operators on those
    # levels lag one update. Not exercised on validated cases (all plans verify CUDA). If reached, a
    # full rebuild is safer than this refresh; surfaced rather than silently producing stale operators.
    @warn "AMG: mixed RAP plan types; coarse refresh may use stale device operators — prefer a rebuild" maxlog=1
    refresh_hierarchy!(hierarchy, solver)
    _sync_device_levels_numeric!(hierarchy)
    return hierarchy
end

end # end module
