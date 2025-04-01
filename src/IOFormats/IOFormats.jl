module IOFormats

using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.Discretise
using KernelAbstractions
using LinearAlgebra
using StaticArrays
using Printf

include("VTK/VTK_types.jl")
include("VTK/VTK_writer.jl")
include("VTK/VTK_writer_3D.jl")

include("OpenFOAM/OpenFOAM_types.jl")
include("OpenFOAM/OpenFOAM_writer.jl")

end