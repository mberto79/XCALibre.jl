module Multithread

export AutoTune
export _setup, xcal_foreach

using KernelAbstractions
import AcceleratedKernels as AK
using SparseArrays
using SparseMatricesCSR
using LinearAlgebra

import Base
import LinearAlgebra
import SparseArrays
import KernelAbstractions

include("spmvm.jl")

struct AutoTune end

_setup(backend::CPU, workgroup::AutoTune, ndrange::I) where {I<: Integer} = begin
    (backend, cld(ndrange, Threads.nthreads()), ndrange)
end

_setup(backend, workgroup::I, ndrange::I) where {I<: Integer} = begin
    (backend, workgroup, ndrange)
end

xcal_foreach(func, arr, config) = begin
    hardware = config.hardware
    (; backend, workgroup) = hardware
    ndrange = length(arr)
    backend, workgroup, ndrange = _setup(backend, workgroup, ndrange)
    AK.foreachindex(func, arr, backend, min_elems=workgroup, block_size=workgroup)
end

end # end module