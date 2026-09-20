module Multithread

export AutoTune
export _setup, _dynamic_setup, xcal_foreach

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
include("profiling.jl")

struct AutoTune end

_setup(backend::CPU, workgroup::AutoTune, ndrange::I) where {I<: Integer} = begin
    (backend, cld(ndrange, Threads.nthreads()), ndrange)
end

_setup(backend, workgroup::I, ndrange::I) where {I<: Integer} = begin
    (backend, workgroup, ndrange)
end

# Counterpart to _setup for kernels whose size changes between launches. Sizes given to a
# kernel constructor become type parameters, recompiling the kernel for every distinct
# size, so these are returned as launch keywords and the kernel is built from the backend.
_dynamic_setup(backend, workgroup, ndrange) = begin
    _, wg, nd = _setup(backend, workgroup, ndrange)
    (workgroupsize=wg, ndrange=nd)
end

xcal_foreach(func, arr, config) = begin
    hardware = config.hardware
    (; backend, workgroup) = hardware
    ndrange = length(arr)
    backend, workgroup, ndrange = _setup(backend, workgroup, ndrange)
    # AK.foreachindex(func, arr, backend, min_elems=workgroup, block_size=workgroup)
    AK.foreachindex(func, arr, min_elems=workgroup, block_size=workgroup)
end

end # end module