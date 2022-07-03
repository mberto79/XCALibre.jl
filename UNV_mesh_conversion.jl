using Plots
using StaticArrays
using BenchmarkTools

using FVM_1D

using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.UNV

points, elements, boundaries = load("unv_sample_meshes/quad.unv", Int64, Float64)

@time nodes, faces = build_mesh("unv_sample_meshes/quad.unv")

scatter(nodes[1:40])