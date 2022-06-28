module UNV

using StaticArrays
using Printf

using FVM_1D.Mesh2D

include("UNV_0_types.jl")
include("UNV_1_loader.jl")
include("UNV_2_builder.jl")

end