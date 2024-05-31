module Discretise

using Accessors
using LinearAlgebra
using SparseArrays
using StaticArrays
using Adapt
using CUDA
using KernelAbstractions
using Atomix
using GPUArrays

using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.ModelFramework

include("Discretise_0_types.jl")
include("Discretise_1_schemes.jl")
include("Discretise_2_generated_distretisation.jl")
include("Discretise_3_boundary_conditions.jl")
include("Discretise_4_apply_bcs.jl")

end