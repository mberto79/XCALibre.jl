module Solve

using Adapt
using Krylov
using LinearAlgebra
using Statistics
using SparseArrays
using SparseMatricesCSR
using LinearOperators
using LDLFactorizations
using KernelAbstractions
using Atomix
using ILUZero
import KrylovPreconditioners as KP

# using ThreadedSparseCSR
# ThreadedSparseCSR.multithread_matmul(BaseThreads())
# # ThreadedSparseCSR.multithread_matmul(PolyesterThreads())


using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise

using Reexport

# @reexport using ILUZero


include("Preconditioners/Preconditioners.jl")
include("Smoothers/Smoothers.jl")
include("Solve_1_api.jl")

end