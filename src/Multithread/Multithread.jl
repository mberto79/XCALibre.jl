module Multithread

using KernelAbstractions
using SparseArrays
using SparseMatricesCSR
using LinearAlgebra

import Base
import LinearAlgebra
import SparseArrays
import KernelAbstractions

include("spmvm.jl")

_workgroup(backend, workgroup::Integer, range) = workgroup
_workgroup(backend::CPU, workgroup::Integer, range) = cld(range, Threads.nthreads())
export _workgroup

end # end module