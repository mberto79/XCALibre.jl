module Discretise

using Accessors
using SparseArrays
using StaticArrays
using Adapt
# using CUDA
using KernelAbstractions
using Atomix
using GPUArrays

using LinearAlgebra
using SparseMatricesCSR

using XCALibre.Multithread
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
# # using XCALibre.Energy

include("boundary_conditions/0_definition_macro.jl")

include("Discretise_0_types.jl")
include("Discretise_1_schemes.jl")
include("Discretise_2_generated_distretisation.jl")
include("Discretise_3_boundary_conditions.jl")
include("Discretise_4_assign_boundaries.jl")
include("Discretise_5_apply_bcs.jl")

include("boundary_conditions/dirichlet.jl")
include("boundary_conditions/dirichlet_interpolation.jl")
include("boundary_conditions/dirichletFunction.jl")
include("boundary_conditions/dirichletFunction_interpolation.jl")
include("boundary_conditions/fixedTemperature.jl")
include("boundary_conditions/fixedTemperature_interpolation.jl")
include("boundary_conditions/neumann.jl")
include("boundary_conditions/neumann_interpolation.jl")
include("boundary_conditions/neumannFunction.jl")
include("boundary_conditions/neumannFunction_interpolation.jl")
include("boundary_conditions/periodic.jl")
include("boundary_conditions/periodic_interpolation.jl")
include("boundary_conditions/symmetry.jl")
include("boundary_conditions/symmetry_interpolation.jl")
include("boundary_conditions/wall.jl")
include("boundary_conditions/wall_interpolation.jl")
export adjust_boundary!

end