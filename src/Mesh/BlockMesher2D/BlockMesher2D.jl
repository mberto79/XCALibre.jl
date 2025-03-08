module BlockMesher2D

using StaticArrays
using LinearAlgebra
using Accessors
using Adapt
using KernelAbstractions

import XCALibre.Mesh as XMesh

include("0_Mesh2_types_old.jl")
include("1_builder_types.jl")
include("2_builder.jl")
include("3_connectivity.jl")
include("4_geometry.jl")
# include("5_elements.jl")
include("6_generate.jl")
include("7_access_functions.jl")

end