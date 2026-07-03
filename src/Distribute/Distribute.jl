module Distribute

using SparseArrays, StaticArrays, Accessors, LinearAlgebra
using MPI, Metis
using KernelAbstractions, Atomix
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
import XCALibre.Solvers: correct_boundary_mass_flux!, nonorthogonal_face_correction,
    _max_courant_number!, update_dt!
import XCALibre.Mesh: _get_float
import XCALibre.ModelFramework: _A, _b, _rowptr, _colval, _nzval, get_phi, get_values
import XCALibre.Solve
import XCALibre.Solve: solve_equation!, solve_system!, residual, setReference!, implicit_relaxation!

include("Distribute_0_types.jl")
include("Distribute_1_partition.jl")
include("Distribute_2_halo.jl")
include("Distribute_3_fields.jl")
include("Distribute_4_linalg.jl")
include("Distribute_5_solvers.jl")
include("Distribute_6_simple.jl")

end # module
