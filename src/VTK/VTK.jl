module VTK

using FVM_1D.Mesh
using FVM_1D.Fields
# using CUDA
using KernelAbstractions
using LinearAlgebra


include("VTK_types.jl")
include("VTK_writer.jl")
include("VTK_writer_3D.jl")

end