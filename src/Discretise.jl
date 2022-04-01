module Discretise

using SparseArrays
using StaticArrays

using FVM_1D.Mesh2D

include("Discretise_0_types.jl")
# include("Discretise_1_operations.jl")
include("Discretise_2_boundary_conditions.jl")
include("Discretise_2_operators.jl")
include("Discretise_3_macros.jl")
include("Discretise_4_functions.jl")

end