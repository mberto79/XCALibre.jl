module Multithread

export AutoTune
export _setup, _dynamic_setup, xcal_foreach

using KernelAbstractions
import AcceleratedKernels as AK
using SparseArrays
using SparseMatricesCSR
using LinearAlgebra
import Krylov

import Base
import LinearAlgebra
import SparseArrays
import KernelAbstractions

include("spmvm.jl")
include("xvector.jl")

struct AutoTune end

_setup(backend::CPU, workgroup::AutoTune, ndrange::I) where {I<: Integer} = begin
    (backend, cld(max(ndrange, one(I)), Threads.nthreads()), ndrange)
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

# A kernel built with its launch range becomes a new type per range, so every mesh, patch and rank
# size compiles its own copy; _sized fixes only a configured workgroup and passes the range at launch.
struct SizedLaunch{K,N}
    kernel::K
    ndrange::N
end
(s::SizedLaunch)(args...; ndrange=s.ndrange) = s.kernel(args...; ndrange)

struct AutoSizedLaunch{K,W,N}
    kernel::K
    workgroupsize::W
    ndrange::N
end
(s::AutoSizedLaunch)(args...; ndrange=s.ndrange) = s.kernel(args...; workgroupsize=s.workgroupsize, ndrange)

_sized(f, backend, workgroup::Integer, ndrange) = SizedLaunch(f(backend, workgroup), ndrange)
_sized(f, backend, workgroup, ndrange) = begin
    _, wg, nd = _setup(backend, workgroup, ndrange)
    AutoSizedLaunch(f(backend), wg, nd)
end

xcal_foreach(func, arr, config) = begin
    (; backend, workgroup) = config.hardware
    _xcal_foreach(func, arr, backend, workgroup)
end

_xcal_foreach(func, arr, backend::CPU, workgroup) = begin
    _, workgroup, _ = _setup(backend, workgroup, length(arr))
    AK.foreachindex(func, arr, min_elems=workgroup, block_size=workgroup)
end

# AcceleratedKernels passes the closure to a callee that is not inlined, so its captured
# environment is copied to per-thread local memory. A kernel that calls it directly is
# scalarised instead, which matters because closures here capture whole fields.
_xcal_foreach(func, arr, backend, workgroup) = begin
    kernel! = _xcal_foreach!(backend)
    kernel!(func; _dynamic_setup(backend, workgroup, length(arr))...)
end

@kernel inbounds=true function _xcal_foreach!(func)
    i = @index(Global)
    @inline func(i)
end

end # end module