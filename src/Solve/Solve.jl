module Solve

using Adapt
using Krylov
using Statistics
using SparseArrays
using LinearOperators
# using LDLFactorizations
using KernelAbstractions

using Atomix
using ILUZero
import KrylovPreconditioners as KP

using LinearAlgebra
using SparseMatricesCSR
using Krylov

using XCALibre.Multithread
using XCALibre.Multithread: _sized, AutoSizedLaunch
import XCALibre.Multithread: XVector, _foreach_chunk, _each_chunk_task, _chunk, _MIN_THREADED_WORK
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise

# using Reexport


include("Preconditioners/Preconditioners.jl")
include("Smoothers/Smoothers.jl")
include("Solve_1_Krylov_solvers.jl")
include("Solve_1_api.jl")
include("AMG/AMG.jl")

end
