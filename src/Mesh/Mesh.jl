module Mesh

using StaticArrays
using LinearAlgebra
using Setfield
using Adapt
using KernelAbstractions
using GPUArrays
using StructArrays
# using CUDA, AMDGPU

include("Mesh_0_types.jl")

include("Mesh_1_functions.jl")

end