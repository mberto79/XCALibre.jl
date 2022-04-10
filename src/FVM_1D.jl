module FVM_1D

# using StaticArrays
# using SparseArrays
# using LinearAlgebra
# using Krylov

# include("Fields.jl")
# include("FVM_discretisation.jl")
# include("prototying.jl")

include("Mesh2D/Mesh2D.jl")
include("Mesh2D/Mesh2D_plotting.jl")
include("Discretise.jl")
include("Models.jl")
include("Calculate.jl")
include("Solvers.jl")

end # module
