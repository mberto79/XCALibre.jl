module Solvers

using Accessors
using SparseArrays
using StaticArrays
using Statistics
using Krylov
using LinearOperators
using ProgressMeter
using KernelAbstractions
import AcceleratedKernels as AK
using Atomix
using Adapt

using LinearAlgebra
using SparseMatricesCSR

using XCALibre.Multithread
using XCALibre.Multithread: _sized
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve
using XCALibre.Solve: _index_type
using XCALibre.Calculate
using XCALibre.ModelPhysics
using XCALibre.IOFormats
using XCALibre.Postprocess
using XCALibre.ReferenceFrames

import XCALibre.ModelPhysics as ModelPhysics

# no display object is built when progress output is off
_progress_bar(iterations, show::Bool) = show ? Progress(iterations; dt=1.0, showspeed=true) : nothing

include("Solvers_0_functions.jl")
include("Solvers_1_SIMPLE-MRF.jl")
include("Solvers_1_SIMPLE.jl")
include("Solvers_1_LAPLACE.jl")
include("Solvers_1_CSIMPLE.jl")
include("Solvers_2_PISO.jl")
include("Solvers_2_CPISO.jl")
include("Solvers_4_Godunov.jl")
include("Solvers_5_Multiphase.jl")
include("Solvers_6_potential_flow.jl")
include("Solvers_3_solver_dispatch.jl")
include("Solvers_1_FilmModel.jl")

end
