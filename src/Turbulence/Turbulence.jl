module Turbulence

using Atomix
using KernelAbstractions
using Accessors
using StaticArrays
using LinearAlgebra
using Adapt
using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.ModelFramework
using FVM_1D.Discretise
using FVM_1D.ModelPhysics
using FVM_1D.Solve
using FVM_1D.Calculate
using FVM_1D.VTK

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

export initialise, turbulence!, model2vtk

end # end module