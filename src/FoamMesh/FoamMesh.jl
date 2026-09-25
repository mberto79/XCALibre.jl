module FoamMesh

using StaticArrays
using LinearAlgebra
using Accessors
using Adapt
using Printf
using Statistics

using XCALibre.Multithread
using XCALibre.Mesh

using XCALibre.Mesh: _check_index_capacity, _with_index_capacity
include("FoamMesh_0_types.jl")
include("FoamMesh_1_read.jl")
include("FoamMesh_2_connect.jl")
include("FoamMesh_3_generate.jl")
include("FoamMesh_4_geometry.jl")
include("FoamMesh_5_build.jl")

end # module end