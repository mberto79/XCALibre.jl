module FVM_1D

using StaticArrays
using SparseArrays
using LinearAlgebra
# using Krylov

# include("Fields.jl")
# include("FVM_discretisation.jl")
# include("Models_builder.jl")
# include("prototying.jl")

include("Mesh2D.jl")
include("Mesh2D_plotting.jl")

include("Discretise.jl")

end # module
