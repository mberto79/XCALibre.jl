module FoamMesh

using StaticArrays
using LinearAlgebra
using Accessors
using Adapt
using Printf
using Statistics

using FVM_1D.Mesh

include("FoamMesh_0_read.jl")
include("FoamMesh_1_connect.jl")
include("FoamMesh_2_generate.jl")
include("FoamMesh_3_geometry.jl")
include("FoamMesh_4_build.jl")

end # module end