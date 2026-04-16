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
                                _build_A, _build_opA, _build_sparse_device
import XCALibre.Solve: _tc_sparse_type, _tc_vec_type

_build_A(backend::BACKEND, i, j, v, n) = begin
    A = sparse(i, j, v, n, n)
    SPARSEGPU(A)
end

function _build_sparse_device(::BACKEND,
        rowptr::AbstractVector, colval::AbstractVector, nzval::AbstractVector{Tv},
        m::Int, n::Int) where {Tv}
    rowptr_gpu = adapt(GPUARRAY{Int32}, rowptr)
    colval_gpu = adapt(GPUARRAY{Int32}, colval)
    nzval_gpu  = adapt(GPUARRAY{Tv}, nzval)
    return SPARSEGPU(rowptr_gpu, colval_gpu, nzval_gpu, (m, n))
end

_build_opA(A::SPARSEGPU) = KP.KrylovOperator(A)

# Mixed-precision type mappings for AMG hierarchy construction
_tc_sparse_type(::Type{SPARSEGPU{Tv, Ti}}) where {Tv, Ti} = SPARSEGPU{Float32, Ti}
_tc_vec_type(::Type{GPUARRAY{Tv, N, B}}) where {Tv, N, B} = GPUARRAY{Float32, N, B}
@inline _nzval(A::SPARSEGPU) = A.nzVal
@inline _rowptr(A::SPARSEGPU) = A.rowPtr
@inline _colval(A::SPARSEGPU) = A.colVal
@inline get_sparse_fields(A::SPARSEGPU) = begin
    A.nzVal, A.colVal, A.rowPtr
end

# ── cuSPARSE-backed AMG SpMV overrides ───────────────────────────────────────
# The generic AMG kernels use a 1-thread-per-row KA SpMV (~22% warp utilisation
# for 7-nnz/row FVM stencils). These overrides replace them with cuSPARSE's
# vectorised CSR SpMV (cusparseSpMV), which is ~4x faster for typical FVM
# pressure matrices. Chebyshev and the PCG loop benefit automatically since
# they call amg_spmv!/amg_residual! directly.

import XCALibre.Solve: amg_spmv!, amg_spmv_add!, amg_residual!,
                        amg_jacobi_sweep!, amg_l1jacobi_sweep!,
                        amg_jacobi_correction!, amg_rap_update_smooth!,
                        _build_coarse_lu!, _refresh_coarse_lu!,
                        _build_smooth_AP_device!,
                        LevelExtras, _fill_dense_from_sparse!,
                        _MAX_DENSE_LU_N,
                        _AMG_T_RAP_CAST, _AMG_T_RAP_AP, _AMG_T_RAP_RAP
import XCALibre.Multithread: _setup

# ── GPU-resident coarse LU ────────────────────────────────────────────────────────────
# Dense LU kept on GPU to avoid PCIe transfers during the V-cycle hot path.

function _build_coarse_lu!(extras::LevelExtras, A_cpu, ::BACKEND, workgroup)
    Tv = eltype(A_cpu.nzval)
    n  = size(A_cpu, 1)
    lu_dense_cpu     = zeros(Tv, n, n)
    _fill_dense_from_sparse!(lu_dense_cpu, A_cpu)
    lu_dense_gpu     = adapt(GPUARRAY{Tv}, lu_dense_cpu)
    extras.lu_dense  = lu_dense_gpu
    extras.lu_factor = lu!(lu_dense_gpu; check=false)   # CUSOLVER getrf! in-place
    extras.lu_rhs    = CUDA.zeros(Tv, n)
    nothing
end

function _refresh_coarse_lu!(extras::LevelExtras, A_device::SPARSEGPU, ::BACKEND)
    nzval_c = A_device.nzVal
    copyto!(extras.A_cpu.nzval, nzval_c)               # GPU sparse nzval → CPU
    Tv = eltype(extras.lu_rhs)
    n  = size(extras.A_cpu, 1)
    lu_dense_cpu = zeros(Tv, n, n)
    _fill_dense_from_sparse!(lu_dense_cpu, extras.A_cpu)
    copyto!(extras.lu_dense, lu_dense_cpu)             # CPU dense → GPU dense
    extras.lu_factor = lu!(extras.lu_dense; check=false)
    nothing
end

# ── Device AP pre-allocation for smooth_P Galerkin update ────────────────────
# Uploads AP_cpu to device as a Float32 sparse matrix at hierarchy setup time.
# Subsequent Galerkin updates call amg_ap_update! + amg_rp_update! (KA kernels)
# which update AP and Ac in-place — zero GPU memory allocation per update call.
# Float64 AP_cpu (fine level) is cast to Float32 on upload; Float32 AP_cpu
# (coarse levels) is uploaded directly.

function _build_smooth_AP_device!(extras::LevelExtras, AP_cpu, ::BACKEND)
    m, n   = size(AP_cpu)
    rowptr = adapt(GPUARRAY{Int32}, Vector{Int32}(AP_cpu.rowptr))
    colval = adapt(GPUARRAY{Int32}, Vector{Int32}(AP_cpu.colval))
    nzval  = adapt(GPUARRAY{Float32}, Vector{Float32}(AP_cpu.nzval))
    extras.AP_device = SPARSEGPU(rowptr, colval, nzval, (m, n))
    nothing
end

function amg_spmv!(y::GPUARRAY, A::SPARSEGPU, x::GPUARRAY,
                   backend::BACKEND, workgroup)
    LinearAlgebra.mul!(y, A, x)
end

function amg_spmv_add!(y::GPUARRAY, A::SPARSEGPU, x::GPUARRAY,
                        alpha::Number, backend::BACKEND, workgroup)
    LinearAlgebra.mul!(y, A, x, alpha, one(eltype(y)))
end

function amg_residual!(r::GPUARRAY, A::SPARSEGPU, x::GPUARRAY, b::GPUARRAY,
                        backend::BACKEND, workgroup)
    LinearAlgebra.mul!(r, A, x)   # r = Ax  (beta=0: no extra read of r)
    r .= b .- r                   # r = b - Ax  (fused CuArray broadcast)
end

# Split-form Jacobi sweep: cuSPARSE SpMV + element-wise correction.
# x_new is used as a scratch buffer for Ax, then overwritten with the result.
# Both regular Jacobi (Dinv = 1/A[i,i]) and l1-Jacobi (Dinv = 1/‖a_i‖_1)
# share the same correction formula x_new = x + ω·Dinv·(b - Ax).

function amg_jacobi_sweep!(x_new::GPUARRAY, x::GPUARRAY, Dinv::GPUARRAY,
                            A::SPARSEGPU, b::GPUARRAY, omega,
                            backend::BACKEND, workgroup)
    LinearAlgebra.mul!(x_new, A, x)                          # x_new = Ax
    amg_jacobi_correction!(x_new, x, Dinv, b, omega, backend, workgroup)
end

function amg_l1jacobi_sweep!(x_new::GPUARRAY, x::GPUARRAY, Dinv_l1::GPUARRAY,
                              A::SPARSEGPU, b::GPUARRAY, omega,
                              backend::BACKEND, workgroup)
    LinearAlgebra.mul!(x_new, A, x)                              # x_new = Ax
    amg_jacobi_correction!(x_new, x, Dinv_l1, b, omega, backend, workgroup)
end

# ── cuSPARSE SpGEMM for smoothed-P Galerkin update ───────────────────────────
# Replaces the generic KA kernel (_amg_rap_row_smooth!) which has O(nnz²) linear
# search behaviour for multi-nnz P rows. cuSPARSE SpGEMM handles the random
# sparse-sparse access pattern with GPU-native algorithms.
# Column ordering: cuSPARSE produces sorted column indices, matching the setup-time
# CPU SpGEMM in galerkin_product. copyto! is therefore value-position–safe.

function amg_rap_update_smooth!(Lc, L, ::BACKEND, workgroup)
    P  = L.P   # SPARSEGPU{Float32}
    R  = L.R   # SPARSEGPU{Float32}
    A  = L.A   # SPARSEGPU{T}: Float64 at fine level, Float32 at coarse levels
    ex = L.extras

    # ── Sub-timer: A→Float32 cast (fine level only; coarse levels already Float32) ──
    t0 = time_ns()
    A_f32 = if eltype(A.nzVal) === Float32
        A
    else
        # Lazy-init pre-allocated Float32 nzVal buffer; update in-place each call.
        if isnothing(ex.A_f32_nzval)
            ex.A_f32_nzval = similar(A.nzVal, Float32)
        end
        ex.A_f32_nzval .= A.nzVal   # GPU broadcast: Float64→Float32, no allocation
        SPARSEGPU(A.rowPtr, A.colVal, ex.A_f32_nzval, A.dims)
    end
    CUDA.synchronize()
    _AMG_T_RAP_CAST[] += (time_ns() - t0) * 1e-9

    # ── Sub-timer: first SpGEMM A*P ───────────────────────────────────────────────
    t0 = time_ns()
    AP = A_f32 * P    # cusparseSpGEMM: (n × n) × (n × nc)
    CUDA.synchronize()
    _AMG_T_RAP_AP[] += (time_ns() - t0) * 1e-9

    # ── Sub-timer: second SpGEMM R*(AP) ──────────────────────────────────────────
    t0 = time_ns()
    Ac_new = R * AP   # cusparseSpGEMM: (nc × n) × (n × nc)
    CUDA.synchronize()
    _AMG_T_RAP_RAP[] += (time_ns() - t0) * 1e-9

    # Copy updated values into the pre-allocated Lc.A (same sparsity, same sorted order).
    copyto!(Lc.A.nzVal, Ac_new.nzVal)

    # Eagerly return SpGEMM temporaries to CUDA pool; eliminates GC-deferred finalization pauses.
    CUDA.unsafe_free!(Ac_new.nzVal); CUDA.unsafe_free!(Ac_new.colVal); CUDA.unsafe_free!(Ac_new.rowPtr)
    CUDA.unsafe_free!(AP.nzVal);     CUDA.unsafe_free!(AP.colVal);     CUDA.unsafe_free!(AP.rowPtr)
    nothing
end

import XCALibre.Solve: _m, _n, update_preconditioner!

function sparse_array_deconstructor_preconditioners(arr::SPARSEGPU)
    (; colVal, rowPtr, nzVal, dims) = arr
    return colVal, rowPtr, nzVal, dims[1], dims[2]
end

_m(A::SPARSEGPU) = A.dims[1]
_n(A::SPARSEGPU) = A.dims[2]

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

end # end module


