module RANSModels

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

include("RANS_types.jl")
include("RANS_functions.jl")
include("RANS_laminar.jl")
include("RANS_kOmega.jl")

end # end module