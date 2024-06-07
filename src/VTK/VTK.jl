module VTK

using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.ModelPhysics
using FVM_1D.RANSModels
using CUDA
using KernelAbstractions
# using FVM_1D.Discretise
# using FVM_1D.Models
# using FVM_1D.Calculate

include("VTK_writer.jl")
include("VTK_writer_3D.jl")

end