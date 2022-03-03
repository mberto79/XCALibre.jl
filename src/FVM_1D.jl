module FVM_1D

using StaticArrays
using SparseArrays
using LinearAlgebra
# using Krylov

include("Mesh_1D_types.jl")
include("Mesh_1D_display.jl")
include("Mesh_1D_generation.jl")
include("Fields.jl")
include("FVM_discretisation.jl")
include("Models_builder.jl")
include("prototying.jl")

include("Mesh2D.jl")
# include("Mesh2D_plotting.jl")

end # module
