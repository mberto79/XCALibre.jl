module Mesh2D

using StaticArrays
using LinearAlgebra
using Setfield

include("Mesh2D_0_mesh_types.jl")
include("Mesh2D_1_builder_types.jl")
include("Mesh2D_2_builder.jl")
include("Mesh2D_3_connectivity.jl")
include("Mesh2D_4_geometry.jl")
# include("Mesh2D_1_nodes.jl")
# include("Mesh2D_2_elements.jl")
end