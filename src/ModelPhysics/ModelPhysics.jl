module ModelPhysics

using Atomix
using KernelAbstractions
using Accessors
using StaticArrays
using Adapt
import AcceleratedKernels as AK

using LinearAlgebra
using SparseMatricesCSR

using XCALibre.Multithread
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve
using XCALibre.Calculate
using XCALibre.Simulate

include("0_type_definition.jl")
include("1_flow_types.jl")
include("2_fluid_models.jl")
include("2_multiphase_models.jl")
include("3_physics_API.jl")

include("Energy/Energy.jl")

include("2_solid_models.jl")

include("Turbulence/Turbulence.jl")

include("FluidProperties/FluidProperties.jl")


end # end module