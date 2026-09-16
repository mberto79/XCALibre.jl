module Multithread

export AutoTune
export _setup, _patch_launch, xcal_foreach

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

# Sizes passed to a kernel constructor become type parameters, so a kernel built
# once per boundary patch is recompiled for every distinct patch size. Patch
# kernels take the backend alone and receive their sizes through _patch_launch,
# so all patches share a single compiled kernel.
_patch_launch(backend, workgroup, ndrange) = begin
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