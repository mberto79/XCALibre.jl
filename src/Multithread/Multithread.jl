module Multithread

using KernelAbstractions
using LinearAlgebra, SparseArrays
using SparseMatricesCSR

import Base: *
import LinearAlgebra: mul! 

include("spmvm.jl")

end # end module