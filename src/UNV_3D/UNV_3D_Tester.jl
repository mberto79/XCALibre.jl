#UNV3Dtester
using StaticArrays
using LinearAlgebra
using Setfield
using Adapt
using StaticArrays
using Setfield
using Printf

using FVM_1D.Mesh

include("UNV3Dtypes.jl")
include("UNV3Dloader.jl")
include("UNV3Dbuilder.jl")

unvFile="src/UNV/Mesh_tetrasmall.unv"
mesh=build_mesh(unvFile,scale=1.0)

