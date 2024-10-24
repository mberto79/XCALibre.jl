module XCALibre_CUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using XCALibre, Adapt, SparseArrays
import KrylovPreconditioners as KP

const SparseGPU = CUDA.CUSPARSE.CuSparseMatrixCSR

function XCALibre.Mesh._convert_array!(arr, backend::CUDABackend)
    return adapt(CuArray, arr) # using CuArray
end

import XCALibre.ModelFramework: _nzval, _colptr, _rowval, get_sparse_fields, 
                                _build_A, _build_opA

_build_A(backend::CUDABackend, i, j, v, n) = begin
    # idev = adapt(CuArray, i)
    # jdev = adapt(CuArray, j)
    # vdev = adapt(CuArray, v)
    # SparseGPU(idev, jdev, vdev, (n, n))
    A = sparse(i, j, v, n, n)
    SparseGPU(A)
end
_build_opA(A::SparseGPU) = KP.KrylovOperator(A)

@inline _nzval(A::SparseGPU) = A.nzVal
# @inline _colptr(A::SparseGPU) = A.colPtr
# @inline _rowval(A::SparseGPU) = A.rowVal
@inline _colptr(A::SparseGPU) = A.rowPtr
@inline _rowval(A::SparseGPU) = A.colVal
# @inline get_sparse_fields(A::SparseGPU) = begin
#     A.nzVal, A.rowVal, A.colPtr
# end
@inline get_sparse_fields(A::SparseGPU) = begin
    A.nzVal, A.colVal, A.rowPtr
end

import XCALibre.Solve: integer_type, _m, _n

integer_type(A::SparseGPU{Tf,Ti}) where {Tf,Ti} = Ti

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

end