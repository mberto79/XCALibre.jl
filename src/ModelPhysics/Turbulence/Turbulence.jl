# module Turbulence

using Atomix
using KernelAbstractions
using Accessors
using StaticArrays
using LinearAlgebra
using Adapt
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.ModelPhysics
using XCALibre.Solve
using XCALibre.Calculate
using XCALibre.IOFormats

include("turbulence_types.jl")

# RANS models
include("RANS_functions.jl")
include("RANS_tensor_algebra.jl")
include("RANS_laminar.jl")
include("RANS_kOmega.jl")
include("RANS_kOmegaLKE.jl")

# LES models
include("LES_functions.jl")
include("LES_Smagorinsky.jl")

export initialise, turbulence!, save_output

# end # end module