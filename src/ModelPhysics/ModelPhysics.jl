module ModelPhysics

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
using FVM_1D.Solve
using FVM_1D.Calculate

include("0_type_definition.jl")
include("1_flow_types.jl")
include("2_fluid_models.jl")
include("3_physics_API.jl")


end # end module