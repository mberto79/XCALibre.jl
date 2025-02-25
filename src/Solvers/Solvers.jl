module Solvers

using Accessors
using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using Krylov
using LinearOperators
using ProgressMeter
using Printf
# using CUDA
using KernelAbstractions
using Atomix
using Adapt

using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve
using XCALibre.Calculate
using XCALibre.ModelPhysics
using XCALibre.Turbulence
using XCALibre.Energy
using XCALibre.VTK

include("Solvers_0_functions.jl")
include("Solvers_1_SIMPLE.jl")
include("Solvers_1_SIMPLE_comp.jl")
include("Solvers_2_PISO.jl")
include("Solvers_3_solver_dispatch.jl")

end