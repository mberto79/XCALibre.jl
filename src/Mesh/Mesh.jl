module Mesh

using StaticArrays
using LinearAlgebra
using Setfield
using Adapt
using KernelAbstractions
using GPUArrays
using XCALibre.Multithread: _active_backend
# using CUDA, AMDGPU

include("Mesh_0_types.jl")

include("Mesh_1_functions.jl")

include("Mesh_2_reorder.jl")

end