module XCALibre

using Krylov 
export BicgstabSolver, CgSolver, GmresSolver

using KernelAbstractions; export CPU
import Adapt: adapt; export adapt

# using ThreadedSparseCSR
# ThreadedSparseCSR.multithread_matmul(BaseThreads())
# # ThreadedSparseCSR.multithread_matmul(PolyesterThreads())


include("Mesh/Mesh.jl")
include("UNV2/UNV2.jl")
include("UNV3/UNV3.jl")
include("FoamMesh/FoamMesh.jl")
# include("Mesh/Plotting/0_plotting.jl")
include("Fields/Fields.jl")
include("ModelFramework/ModelFramework.jl")
include("Discretise/Discretise.jl")
include("Solve/Solve.jl")
include("Calculate/Calculate.jl")
include("VTK/VTK.jl")
include("ModelPhysics/ModelPhysics.jl")
# include("Turbulence/Turbulence.jl")
# include("Energy/Energy.jl")
include("Simulate/Simulate.jl")
include("Solvers/Solvers.jl")
include("Postprocess/Postprocess.jl")

using Reexport
@reexport using XCALibre.Mesh
@reexport using XCALibre.FoamMesh
# @reexport using XCALibre.Plotting
@reexport using XCALibre.Fields
@reexport using XCALibre.ModelFramework
@reexport using XCALibre.Discretise
@reexport using XCALibre.Solve
@reexport using XCALibre.Calculate
@reexport using XCALibre.ModelPhysics
# @reexport using XCALibre.Turbulence
# @reexport # using XCALibre.Energy
@reexport using XCALibre.Simulate
@reexport using XCALibre.Solvers
@reexport using XCALibre.Postprocess
@reexport using XCALibre.VTK
@reexport using XCALibre.UNV3
@reexport using XCALibre.UNV2

# using PrecompileTools: @setup_workload, @compile_workload


# @setup_workload begin
#         @compile_workload begin
#                 include("../CASE_UNV_BFS.jl")
#         end
# end

end # module
