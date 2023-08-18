module ModelFramework

using Accessors
# using LinearAlgebra
using SparseArrays
# using StaticArrays


using LoopVectorization
# using LinearAlgebra
# using Statistics
using Krylov
# using LinearOperators
# using ProgressMeter

using FVM_1D.Mesh
using FVM_1D.Fields
# using FVM_1D.Discretise

include("ModelFramework_0_types.jl")
include("ModelFramework_1_operations.jl")
include("ModelFramework_2_access_functions.jl")

end