module Mesh

using StaticArrays
using LinearAlgebra
using Setfield

include("Mesh_0_types.jl")
include("Mesh2D/Mesh2D_0_types.jl")
include("Mesh2D/Mesh2D_1_builder_types.jl")
include("Mesh2D/Mesh2D_2_builder.jl")
include("Mesh2D/Mesh2D_3_connectivity.jl")
include("Mesh2D/Mesh2D_4_geometry.jl")
include("Mesh2D/Mesh2D_5_access_functions.jl")
# include("Mesh2D/6_elements.jl")
include("Mesh2D/Mesh2D_7_generate.jl")

end