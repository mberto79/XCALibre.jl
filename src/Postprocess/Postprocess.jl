module Postprocess

using StaticArrays

using LinearAlgebra
using SparseMatricesCSR
# using ThreadedSparseCSR

using Adapt
using XCALibre.Multithread
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.Calculate
using XCALibre.Discretise
using XCALibre.ModelPhysics
using XCALibre.Solve
# using XCALibre.Turbulence

include("Postprocess_functions.jl")
include("Postprocess_0_field_average.jl")
include("Postprocess_1_field_rms.jl")
include("Postprocess_2_reynolds_stress_tensor.jl")
include("Postprocess_3_Q-criterion.jl")

end