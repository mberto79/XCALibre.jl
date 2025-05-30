module XCALibre_oneAPIExt

if isdefined(Base, :get_extension)
    using oneAPI
else
    using ..oneAPI
end

using XCALibre, Adapt, SparseArrays, SparseMatricesCSR, KernelAbstractions
using LinearAlgebra, LinearOperators
import KrylovPreconditioners as KP

const SPARSEGPU = oneAPI.oneMKL.oneSparseMatrixCSR
const BACKEND = oneAPIBackend
const GPUARRAY = oneArray
# using oneAPI.oneMKL: ilu02, ilu02!, ic02, ic02!

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
@inline get_sparse_fields(A::SPARSEGPU) = begin
    A.nzVal, A.colVal, A.rowPtr
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
    error("Not yet implemented for type $(typeof(A))")
end

# ILU0GPU

Preconditioner{ILU0GPU}(A::SPARSEGPU) = begin
    error("Not yet implemented for type $(typeof(A))")
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


