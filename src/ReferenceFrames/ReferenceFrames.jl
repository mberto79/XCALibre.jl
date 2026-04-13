module ReferenceFrames

using Adapt
using LinearAlgebra
using KernelAbstractions
using StaticArrays

using XCALibre.Multithread
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve
using XCALibre.Calculate
using XCALibre.ModelPhysics
using XCALibre.IOFormats
using XCALibre.Postprocess

include("ReferenceFrames_0_type_definition.jl")
include("ReferenceFrames_1_output_functions.jl")
include("ReferenceFrames_1_zone_functions.jl")
    
end