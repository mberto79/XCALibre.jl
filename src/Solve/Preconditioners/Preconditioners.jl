using Adapt
using SparseArrays
using KernelAbstractions
using CUDA

include("preconditioners_0_types.jl")
include("preconditioners_1_DILU.jl")
include("preconditioners_2_functions.jl")
