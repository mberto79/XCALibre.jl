module FVM_1D

include("Mesh/Mesh.jl")
include("Mesh/Plotting/0_plotting.jl")
include("Fields/Fields.jl")
include("Discretise/Discretise.jl")
include("Model/Models.jl")
include("Calculate/Calculate.jl")
include("Solve/Solvers.jl")
include("VTK/VTK.jl")
include("UNV/UNV.jl")

using Reexport
@reexport using FVM_1D.Mesh
@reexport using FVM_1D.Plotting
@reexport using FVM_1D.Fields
@reexport using FVM_1D.Discretise
@reexport using FVM_1D.Calculate
@reexport using FVM_1D.Models
@reexport using FVM_1D.Solvers
@reexport using FVM_1D.VTK
@reexport using FVM_1D.UNV

end # module
