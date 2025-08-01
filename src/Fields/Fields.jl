module Fields

import XCALibre: CONFIG

using Adapt
using KernelAbstractions
# using CUDA
using LinearAlgebra
using SparseArrays
using StaticArrays
import KernelAbstractions as KA

using XCALibre.Multithread
using XCALibre.Mesh

include("Fields_0_types.jl")

end