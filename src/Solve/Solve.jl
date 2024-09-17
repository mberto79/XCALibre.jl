module Solve

using Adapt
using Krylov
using LinearAlgebra
using Statistics
using SparseArrays
using LinearOperators
using LDLFactorizations
using KernelAbstractions
using Atomix
using ILUZero

using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise

using Reexport

# @reexport using ILUZero


include("Preconditioners/Preconditioners.jl")
include("Solve_1_api.jl")

end