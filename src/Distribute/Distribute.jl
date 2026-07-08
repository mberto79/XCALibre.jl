module Distribute

using SparseArrays, StaticArrays, Accessors, LinearAlgebra, Serialization
using MPI, Metis
using Printf
using Logging
using KernelAbstractions, Atomix
import Adapt
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.Multithread
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve
using XCALibre.ModelPhysics
using XCALibre.Calculate
using XCALibre.Solvers
import XCALibre.Fields: initialise!
import XCALibre.Solvers: global_max, _base_mesh
import XCALibre.Mesh: _get_float
import XCALibre.ModelFramework: _A, _b, _rowptr, _colval, _nzval, get_phi, get_values
import XCALibre.Solve
import XCALibre.Solve: solve_equation!, solve_system!, residual, setReference!,
    implicit_relaxation!, make_symmetric!
using XCALibre.IOFormats
import XCALibre.IOFormats: initialise_writer, write_results, copy_to_cpu, copy_scalarfield_to_cpu, get_data
import XCALibre.Mesh: _get_backend

include("Distribute_0_types.jl")
include("Distribute_1_partition.jl")
include("Distribute_2_halo.jl")
include("Distribute_3_fields.jl")
include("Distribute_4_linalg.jl")
include("Distribute_5_solvers.jl")
include("Distribute_7_io.jl")

end # module
