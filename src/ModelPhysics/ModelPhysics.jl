module ModelPhysics

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
using XCALibre.Solve
using XCALibre.Calculate

include("0_type_definition.jl")
include("1_flow_types.jl")
include("2_fluid_models.jl")
include("3_physics_API.jl")

include("Energy/Energy.jl")

end # end module