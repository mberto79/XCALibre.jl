module Solvers

using Accessors
using SparseArrays
using StaticArrays
using Statistics
using Krylov
using LinearOperators
using ProgressMeter
using KernelAbstractions
using Atomix
using Adapt

using LinearAlgebra
using SparseMatricesCSR

using XCALibre.Multithread
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve
using XCALibre.Calculate
using XCALibre.ModelPhysics
using XCALibre.IOFormats

include("Solvers_0_functions.jl")
include("Solvers_1_SIMPLE.jl")
include("Solvers_1_CSIMPLE.jl")
include("Solvers_2_PISO.jl")
include("Solvers_2_CPISO.jl")
include("Solvers_3_solver_dispatch.jl")

end