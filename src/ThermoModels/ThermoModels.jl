module ThermoModels

using Accessors
using StaticArrays
using LinearAlgebra

using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.ModelFramework
using FVM_1D.Discretise
using FVM_1D.Solve
using FVM_1D.Calculate

# include("ThermoModels_0_types.jl")
# include("ThermoModels_constant_rho.jl")
include("ThermoModels_perfect_gas.jl")

end # end module