module ModelPhysics

using Atomix
using KernelAbstractions
using Accessors
using StaticArrays
using Adapt
import AcceleratedKernels as AK

using LinearAlgebra
using SparseMatricesCSR

using XCALibre.Multithread
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve
using XCALibre.Calculate
using XCALibre.Simulate

include("0_type_definition.jl")
include("1_flow_types.jl")
include("2_multiphase_sources.jl")
include("2_fluid_models.jl")
include("2_thermophysical_models.jl")
include("2_phase_change_models.jl")
# Needs the `Const*`/`IdealGas` models it extends, and the saturation-model
# supertype from the phase change file, so it follows both.
include("2_tabulated_properties.jl")
include("2_wall_boiling_models.jl")
include("2_viscosity_models.jl")
include("3_physics_API.jl")

include("Energy/Energy.jl")

include("2_solid_models.jl")

include("Turbulence/Turbulence.jl")

include("FluidProperties/FluidProperties.jl")

# AFTER FluidProperties: the `PengRobinson(fluid)` convenience constructor and
# the acentric-factor table dispatch on `H2`/`H2_para`/`N2`, which are defined
# there. The EOS itself depends only on 2_fluid_models and 2_thermophysical.
include("2_peng_robinson.jl")

end # end module