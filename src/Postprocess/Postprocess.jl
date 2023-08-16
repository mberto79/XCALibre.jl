module Postprocess

using LinearAlgebra
using StaticArrays
# using LoopVectorization
# using SparseArrays

using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.Calculate
using FVM_1D.Discretise

include("Postprocess_forces.jl")

end