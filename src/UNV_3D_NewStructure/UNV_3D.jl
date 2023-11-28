using StaticArrays
using LinearAlgebra
using Setfield
using Adapt
using Printf

include("UNV_3D_NS_Types.jl")
include("UNV_3D_NS_Reader.jl")
include("UNV_3D_NS_Builder.jl")

unv_mesh="src/UNV_3D_NewStructure/tetra_singlecell.unv"

build_mesh3D(unv_mesh)