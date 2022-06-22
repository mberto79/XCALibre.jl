module Calculate

using LinearAlgebra
using StaticArrays
using LoopVectorization
# using SparseArrays

using FVM_1D.Mesh2D
using FVM_1D.Discretise

include("Calculate_0_types.jl")
include("Calculate_1_interpolation.jl")
include("Calculate_1_orthogonality_correction.jl")
include("Calculate_2_operators.jl")

end