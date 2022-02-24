module FVM_1D

using StaticArrays
using SparseArrays
using LinearAlgebra
# using Krylov

include("Mesh_1D_types.jl")
include("Mesh_1D_display.jl")
include("Mesh_1D_generation.jl")
include("Fields.jl")
include("Models_builder.jl")
include("FVM_discretisation.jl")
include("prototying.jl")

end # module
