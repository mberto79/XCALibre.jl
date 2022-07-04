using Plots
using StaticArrays
using BenchmarkTools

using FVM_1D

using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.UNV

points, elements, boundaries = load("unv_sample_meshes/quad.unv", Int64, Float64)

@time nodes, faces, cells = build_mesh("unv_sample_meshes/quad.unv")

scatter(nodes)

fig = plot()
for fID ∈ 1:220
    p1 = nodes[faces[fID].nodesID[1]].coords
    p2 = nodes[faces[fID].nodesID[2]].coords
    x = [p1[1], p2[1]]
    y = [p1[2], p2[2]]
    plot!(fig, x,y, legend=false)
end
@show fig