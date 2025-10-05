# module ModelFramework

# using XCALibre

# using Accessors
# using SparseArrays
# using StaticArrays
# using Adapt
# using LinearOperators

# using LinearAlgebra
# using SparseMatricesCSR

# using Krylov
# import KrylovPreconditioners as KP
# using KernelAbstractions

# using XCALibre.Multithread
using XCALibre.Mesh
# using XCALibre.Fields

include("ModelFramework_0_types.jl")
include("ModelFramework_1_operations.jl")
include("ModelFramework_2_access_functions.jl")

# end