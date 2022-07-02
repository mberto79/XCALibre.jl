using Plots
using StaticArrays
using BenchmarkTools

using FVM_1D

using FVM_1D.Mesh2D
using FVM_1D.UNV

points, elements, boundaries = load("unv_sample_meshes/quad.unv", Int64, Float64)

@time nodes = build_mesh("unv_sample_meshes/quad.unv")