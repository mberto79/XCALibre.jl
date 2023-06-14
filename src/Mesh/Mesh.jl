module Mesh

using StaticArrays
using LinearAlgebra
using Setfield

include("0_mesh_types.jl")
include("Mesh2D/0_types.jl")
include("Mesh2D/1_builder_types.jl")
include("Mesh2D/2_builder.jl")
include("Mesh2D/3_connectivity.jl")
include("Mesh2D/4_geometry.jl")
include("Mesh2D/5_access_functions.jl")
# include("Mesh2D/6_elements.jl")
include("Mesh2D/7_generate.jl")

end