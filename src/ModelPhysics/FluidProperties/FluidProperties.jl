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

include("ThermalConductivity/thermal_conductivity_H2.jl")
include("ThermalConductivity/thermal_conductivity_N2.jl")

include("Viscosity/high_fidelity_mu_H2.jl")
include("Viscosity/high_fidelity_mu_N2.jl")

include("surface_tension.jl")
