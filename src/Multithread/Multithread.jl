module Multithread

export _setup

using KernelAbstractions
using SparseArrays
using SparseMatricesCSR
using LinearAlgebra

import Base
import LinearAlgebra
import SparseArrays
import KernelAbstractions

include("spmvm.jl")

_setup(backend::CPU, workgroup::I, ndrange::I) where {I<: Integer} = begin
    # (backend, cld(ndrange, Threads.nthreads()) + one(I), ndrange)
    (backend, cld(ndrange, Threads.nthreads()), ndrange)
end

_setup(backend, workgroup::I, ndrange::I) where {I<: Integer} = begin
    (backend, workgroup, ndrange)
end

end # end module