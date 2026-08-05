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
using XCALibre.ModelPhysics
using XCALibre.Solve
using XCALibre.IOFormats
using XCALibre.Calculate

include("HelmholtzEnergy/HelmholtzFunctions.jl")
include("HelmholtzEnergy/Helmholtz_H2.jl")
include("HelmholtzEnergy/Helmholtz_N2.jl")

include("ThermalConductivity/thermal_conductivity_H2.jl")
include("ThermalConductivity/thermal_conductivity_N2.jl")

include("Viscosity/high_fidelity_mu_H2.jl")
include("Viscosity/high_fidelity_mu_N2.jl")

include("surface_tension.jl")

include("HighFidelity_Closure.jl")

# Builds the kernel-safe property tables consumed by `TabulatedEos` and friends.
# Must come last: it needs the EOS, viscosity, conductivity and saturation
# routines above, as well as the model types from `2_tabulated_properties.jl`.
include("property_tables.jl")
