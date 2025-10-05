# module Multithread

export AutoTune
export _setup

# using KernelAbstractions
# using SparseArrays
# using SparseMatricesCSR
# using LinearAlgebra

# import Base
# import LinearAlgebra
# import SparseArrays
# import KernelAbstractions

include("spmvm.jl")

struct AutoTune end

@inline _setup(backend::CPU, workgroup::AutoTune, ndrange::I) where {I<: Integer} = begin
    # wrkgp = cld(ndrange, Threads.nthreads())
    wrkgp = 2^floor(I, log2(ndrange/Threads.nthreads()))
    (backend, wrkgp, ndrange)
end

_setup(backend, workgroup::I, ndrange::I) where {I<: Integer} = begin
    (backend, workgroup, ndrange)
end

# end # end module