module FVM_1D_AMDExt

if isdefined(Base, :get_extension)
    using AMDGPU
else
    using ..AMDGPU
end

using FVM_1D, Adapt

const SparseGPU = AMDGPU.rocSPARSE.ROCSparseMatrixCSC

function FVM_1D.Mesh._convert_array!(arr, backend::CUDABackend)
    return adapt(ROCArray, arr) # using ROCArray
end

import FVM_1D.ModelFramework: _nzval, _colptr, _rowval, get_sparse_fields
@inline _nzval(A::SparseGPU) = A.nzVal
@inline _colptr(A::SparseGPU) = A.colPtr
@inline _rowval(A::SparseGPU) = A.rowVal
@inline get_sparse_fields(A::SparseGPU) = begin
    A.nzVal, A.rowVal, A.colPtr
end

import FVM_1D.Solve: integer_type, _m, _n

integer_type(A::SparseGPU{Tf,Ti}) where {Tf,Ti} = Ti

function sparse_array_deconstructor_preconditioners(arr::SparseGPU)
    (; rowVal, colPtr, nzVal, dims) = arr
    return rowVal, colPtr, nzVal, dims[1], dims[2]
end

_m(A::SparseGPU) = A.dims[1]
_n(A::SparseGPU) = A.dims[2]

end