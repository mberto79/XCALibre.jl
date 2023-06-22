module Discretise

using Accessors
using LinearAlgebra
using SparseArrays
using StaticArrays

using FVM_1D.Mesh
using FVM_1D.Fields

include("0_types.jl")
include("1_operations.jl")
# include("Discretise_2_boundary_conditions.jl")
# include("Discretise_2_operators.jl")
# include("Discretise_3_macros.jl")
# include("Discretise_4_functions.jl")

end