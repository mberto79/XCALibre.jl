module Preprocess

using StaticArrays

using LinearAlgebra
using SparseMatricesCSR
# using ThreadedSparseCSR

using Adapt
using Atomix
using KernelAbstractions
import KernelAbstractions as KA
using XCALibre.Multithread
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.Calculate
using XCALibre.Discretise
using XCALibre.ModelPhysics
# using XCALibre.Turbulence

include("setFields.jl")

end
