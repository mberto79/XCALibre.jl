module XCALibre_CUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using XCALibre, Adapt, SparseArrays, SparseMatricesCSR, KernelAbstractions
using LinearAlgebra, LinearOperators
import KrylovPreconditioners as KP

const SparseGPU = CUDA.CUSPARSE.CuSparseMatrixCSR

function XCALibre.Mesh._convert_array!(arr, backend::CUDABackend)
    return adapt(CuArray, arr) # using CuArray
end

import XCALibre.ModelFramework: _nzval, _rowptr, _colval, get_sparse_fields, 
                                _build_A, _build_opA

_build_A(backend::CUDABackend, i, j, v, n) = begin

    A = sparse(i, j, v, n, n)
    SparseGPU(A)
end

_build_opA(A::SparseGPU) = KP.KrylovOperator(A)
@inline _nzval(A::SparseGPU) = A.nzVal
@inline _rowptr(A::SparseGPU) = A.rowPtr
@inline _colval(A::SparseGPU) = A.colVal
@inline get_sparse_fields(A::SparseGPU) = begin
    A.nzVal, A.colVal, A.rowPtr
end

import XCALibre.Solve: _m, _n, update_preconditioner!

# function sparse_array_deconstructor_preconditioners(arr::SparseGPU)
#     (; rowVal, colPtr, nzVal, dims) = arr
#     return rowVal, colPtr, nzVal, dims[1], dims[2]
# end

function sparse_array_deconstructor_preconditioners(arr::SparseGPU)
    (; colVal, rowPtr, nzVal, dims) = arr
    return colVal, rowPtr, nzVal, dims[1], dims[2]
end

_m(A::SparseGPU) = A.dims[1]
_n(A::SparseGPU) = A.dims[2]

# DILU Preconditioner (hybrid implementation for now)

Preconditioner{DILU}(Agpu::SparseGPU{F,I}) where {F,I} = begin
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

update_preconditioner!(P::Preconditioner{DILU,M,PT,S},  mesh, config) where {M,PT,S} =
begin
    KernelAbstractions.copyto!(CPU(), P.storage.A.nzval, P.A.nzVal)
    update_dilu_diagonal!(P, mesh, config)
    nothing
end

# IC0GPU

# Preconditioner{IC0GPU}(A::AbstractSparseArray{F,I}) where {F,I} = begin
#     backend = get_backend(A)
#     m, n = size(A)
#     m == n || throw("Matrix not square")
#     # S = _convert_array!(zeros(m), backend)
#     S = zero(I)
#     P = KP.kp_ic0(A)
#     Preconditioner{IC0GPU,typeof(A),typeof(P),typeof(S)}(A,P,S)
# end

# function update_preconditioner!(P::Preconditioner{ILU0GPU,M,PT,S}, mesh, config) where {M<:AbstractSparseArray,PT,S}
#     KP.update!(P.P, P.A)
#     nothing
# end

# IC0GPU NEW

struct IC0GPUStorage{A,B,C}
    P::A
    L::B 
    U::C
end

Preconditioner{IC0GPU}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    # S = _convert_array!(zeros(m), backend)
    # S = zero(I)
    # P = KP.kp_ic0(A)
    PS = CUDA.CUSPARSE.ic02(A)
    # L = LowerTriangular(PS)
    L = KP.TriangularOperator(PS, 'L', 'N', nrhs=1, transa='N')
    # U = LowerTriangular(PS)'
    U = KP.TriangularOperator(PS, 'L', 'N', nrhs=1, transa='T')
    S = IC0GPUStorage(PS,L,U)

    T = eltype(A.nzVal)
    z = CUDA.zeros(T, n)

    P = LinearOperator(T, n, n, true, true, (y, x) -> ldiv_ic0!(S, x, y, z))

    Preconditioner{IC0GPU,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

function ldiv_ic0!(S, x, y, z)
    ldiv!(z, S.L, x)   # Forward substitution with L
    ldiv!(y, S.U, z)  # Backward substitution with Lᴴ
    return y
end

update_preconditioner!(P::Preconditioner{IC0GPU,M,PT,S},  mesh, config) where {M<:SparseGPU,PT,S} = 
begin
    # KP.update!(P.P, P.A)
    P.storage.P.nzVal .= P.A.nzVal
    CUDA.CUSPARSE.ic02!(P.storage.P)
    KP.update!(P.storage.L, P.storage.P)
    KP.update!(P.storage.U, P.storage.P)
    nothing
end

import LinearAlgebra.ldiv!, LinearAlgebra.\
export ldiv!

ldiv!(x::CuArray, P::DILUprecon{M,V,VI}, b) where {M<:AbstractSparseArray,V,VI} =
begin
    xcpu = Vector(x)
    bcpu = Vector(b)
    XCALibre.Solve.forward_substitution!(xcpu, P, bcpu)
    XCALibre.Solve.backward_substitution!(xcpu, P, xcpu)
    KernelAbstractions.copyto!(CUDABackend(), x, xcpu)
end

end # end module


