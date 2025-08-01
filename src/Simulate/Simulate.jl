module Simulate

import XCALibre: CONFIG, get_configuration

using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using Adapt
using KernelAbstractions

using LinearAlgebra
using SparseMatricesCSR
using XCALibre.Multithread

include("Simulate_0_types.jl")

end