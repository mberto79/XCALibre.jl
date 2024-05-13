module ModelFramework

using Accessors
using SparseArrays
using Adapt

using Krylov
using KernelAbstractions
using CUDA

using FVM_1D.Mesh
using FVM_1D.Fields

include("ModelFramework_0_types.jl")
include("ModelFramework_1_operations.jl")
include("ModelFramework_2_access_functions.jl")

end