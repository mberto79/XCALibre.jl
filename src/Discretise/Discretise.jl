module Discretise

using Accessors
using LinearAlgebra
using SparseArrays
using StaticArrays

using FVM_1D.Mesh
using FVM_1D.Fields

include("0_types.jl")
include("1_operations.jl")
include("2_operators.jl")
include("3_generated_distretisation.jl")
include("4_boundary_conditions.jl")
# include("Discretise_3_macros.jl")
include("5_functions.jl")

end