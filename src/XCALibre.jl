module XCALibre

# using Krylov 
# export Bicgstab(), Cg(), Gmres()

using KernelAbstractions; export CPU
import Adapt: adapt; export adapt


include("Multithread/Multithread.jl")
include("Mesh/Mesh.jl")
include("UNV2/UNV2.jl")
include("UNV3/UNV3.jl")
include("FoamMesh/FoamMesh.jl")
include("Fields/Fields.jl")
include("ModelFramework/ModelFramework.jl")
include("Discretise/Discretise.jl")
include("Solve/Solve.jl")
include("Simulate/Simulate.jl")
include("Calculate/Calculate.jl")
include("IOFormats/IOFormats.jl")
include("ModelPhysics/ModelPhysics.jl")
include("Postprocess/Postprocess.jl")
include("ReferenceFrames/ReferenceFrames.jl")
include("Solvers/Solvers.jl")
include("Distribute/Distribute.jl")
include("Preprocess/Preprocess.jl")
include("Mesh/BlockMesher2D/BlockMesher2D.jl")

using Reexport
@reexport using XCALibre.Multithread
@reexport using XCALibre.Mesh
@reexport using XCALibre.FoamMesh
@reexport using XCALibre.Fields
@reexport using XCALibre.ModelFramework
@reexport using XCALibre.Discretise
@reexport using XCALibre.Solve
@reexport using XCALibre.Calculate
@reexport using XCALibre.ModelPhysics
@reexport using XCALibre.Simulate
@reexport using XCALibre.Postprocess
@reexport using XCALibre.ReferenceFrames
@reexport using XCALibre.Solvers
@reexport using XCALibre.Distribute
@reexport using XCALibre.Preprocess
@reexport using XCALibre.IOFormats
@reexport using XCALibre.UNV3
@reexport using XCALibre.UNV2
@reexport using XCALibre.BlockMesher2D

using StaticArrays, LinearAlgebra, SparseMatricesCSR, SparseArrays, LinearOperators
using ProgressMeter, Printf, Adapt



end # module
