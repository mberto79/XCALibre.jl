module Fields

using Adapt
using CUDA
using LinearAlgebra
using SparseArrays
using StaticArrays

using FVM_1D.Mesh

include("Fields_0_types.jl")
include("Fields_1_GPU_Conversion.jl")

end