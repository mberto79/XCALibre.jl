module Solvers

using LoopVectorization
using LinearAlgebra
using LinearOperators
using Krylov
using ILUZero
# using IncompleteLU

using FVM_1D.Mesh2D
using FVM_1D.Discretise
using FVM_1D.Models
using FVM_1D.Calculate

include("Solvers_1_api.jl")
# include("Solvers_2_SIMPLE.jl")

end