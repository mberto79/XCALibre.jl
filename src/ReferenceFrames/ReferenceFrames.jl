module ReferenceFrames

using Adapt
using LinearAlgebra
using KernelAbstractions

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

include("0_rotatingFrame_type_definitions.jl")
include("1_output_functions.jl")
include("1_initialise_rotating_frame.jl")
include("1_MRF_zone_maker.jl")
    
end