module Distribute

using SparseArrays, StaticArrays, Accessors, LinearAlgebra
using MPI, Metis
using KernelAbstractions, Atomix
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.Multithread
import XCALibre.Fields: initialise!

include("Distribute_0_types.jl")
include("Distribute_1_partition.jl")
include("Distribute_2_halo.jl")
include("Distribute_3_fields.jl")
include("Distribute_4_linalg.jl")

end # module
