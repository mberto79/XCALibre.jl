module XCALibre_CUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using XCALibre, Adapt, SparseArrays, SparseMatricesCSR, KernelAbstractions
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

import XCALibre.Solve: _m, _n

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

update_preconditioner!(P::Preconditioner{DILU,M,PT,S},  mesh, config) where {M<:SparseGPU,PT,S} =
begin
    KernelAbstractions.copyto!(CPU(), P.storage.A.nzval, P.A.nzVal)
    update_dilu_diagonal!(P, mesh, config)
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


