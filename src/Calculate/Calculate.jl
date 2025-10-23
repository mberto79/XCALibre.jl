module Calculate

using StaticArrays
using SparseArrays
using Accessors
using Adapt
using Atomix
using KernelAbstractions
using GPUArrays

using LinearAlgebra
using SparseMatricesCSR

using XCALibre.Multithread
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve
using XCALibre.Simulate


include("Calculate_0_gradient.jl")
include("Calculate_0_gradient_cell_limiter.jl")
include("Calculate_0_gradient_face_limiter.jl")
include("Calculate_0_gradient_mface_limiter.jl")
include("Calculate_0_divergence.jl")
include("Calculate_1_green_gauss.jl")
include("Calculate_2_interpolation.jl")
include("Calculate_3_orthogonality_correction.jl")
include("Calculate_4_wall_distance.jl")
include("Calculate_5_surface_normal_gradient.jl")

end