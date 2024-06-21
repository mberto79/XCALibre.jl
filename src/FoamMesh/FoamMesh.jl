module FoamMesh

using StaticArrays
using LinearAlgebra
using Accessors
using Adapt
using Printf
using Statistics

using FVM_1D.Mesh

include("FoamMesh_0_read.jl")
include("FoamMesh_1_connectivity.jl")
include("FoamMesh_2_geometry.jl")

end # module end