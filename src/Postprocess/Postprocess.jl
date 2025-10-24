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
include("Postprocess_0_time_average.jl")
include("Postprocess_1_rms.jl")
include("Postprocess_2_reynolds_stress_tensor.jl")

end