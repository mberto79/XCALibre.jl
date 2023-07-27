module Solve

using LoopVectorization
using LinearAlgebra
using Statistics
using Krylov
using LinearOperators
using ProgressMeter

using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.Discretise
using FVM_1D.Preconditioners
using FVM_1D.Calculate

include("Solve_1_api.jl")
include("Solve_2_SIMPLE.jl")

end