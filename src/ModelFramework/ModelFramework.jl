module ModelFramework

using Accessors
using SparseArrays
using StaticArrays
using Adapt
using LinearOperators

using Krylov
import KrylovPreconditioners as KP
using KernelAbstractions

using XCALibre.Mesh
using XCALibre.Fields

include("ModelFramework_0_types.jl")
include("ModelFramework_1_operations.jl")
include("ModelFramework_2_access_functions.jl")

end