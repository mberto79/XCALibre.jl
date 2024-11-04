module XCALibre_AMDExt

if isdefined(Base, :get_extension)
    using AMDGPU
else
    using ..AMDGPU
end

using XCALibre, Adapt, SparseArrays
import KrylovPreconditioners as KP

const SparseGPU = AMDGPU.rocSPARSE.ROCSparseMatrixCSC

function XCALibre.Mesh._convert_array!(arr, backend::CUDABackend)
    return adapt(ROCArray, arr) # using ROCArray
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

import XCALibre.Solve: integer_type, _m, _n

integer_type(A::SparseGPU{Tf,Ti}) where {Tf,Ti} = Ti

function sparse_array_deconstructor_preconditioners(arr::SparseGPU)
    (; rowVal, colPtr, nzVal, dims) = arr
    return rowVal, colPtr, nzVal, dims[1], dims[2]
end

_m(A::SparseGPU) = A.dims[1]
_n(A::SparseGPU) = A.dims[2]

end # end module