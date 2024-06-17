module UNV2

using StaticArrays
using LinearAlgebra
using Setfield
using Printf
# using CUDA

# using FVM_1D.Mesh

include("UNV2_0_types.jl")
include("UNV2_1_loader.jl")
include("UNV2_2_geometry.jl")
include("UNV2_3_builder.jl")

end