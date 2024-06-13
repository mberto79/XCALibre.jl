module FVM_1D

using Krylov 
export BicgstabSolver, CgSolver

using KernelAbstractions; export CPU

include("UNV2/UNV2.jl")
include("Mesh/Mesh.jl")
# include("Mesh/Plotting/0_plotting.jl")
include("Fields/Fields.jl")
include("ModelFramework/ModelFramework.jl")
include("Discretise/Discretise.jl")
include("Solve/Solve.jl")
include("Calculate/Calculate.jl")
include("ModelPhysics/ModelPhysics.jl")
include("VTK/VTK.jl")
include("RANSModels/RANSModels.jl")
include("Simulate/Simulate.jl")
include("Solvers/Solvers.jl")
include("Postprocess/Postprocess.jl")
include("UNV3/UNV3.jl")

using Reexport
@reexport using FVM_1D.Mesh
# @reexport using FVM_1D.Plotting
@reexport using FVM_1D.Fields
@reexport using FVM_1D.ModelFramework
@reexport using FVM_1D.Discretise
@reexport using FVM_1D.Solve
@reexport using FVM_1D.Calculate
@reexport using FVM_1D.ModelPhysics
@reexport using FVM_1D.RANSModels
@reexport using FVM_1D.Simulate
@reexport using FVM_1D.Solvers
@reexport using FVM_1D.Postprocess
@reexport using FVM_1D.VTK
@reexport using FVM_1D.UNV3
@reexport using FVM_1D.UNV2

end # module
