using Plots
using StaticArrays
using BenchmarkTools

using FVM_1D

using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.UNV

@time points, elements, boundaryElements = load(
    "unv_sample_meshes/quad.unv", Int64, Float64)

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm
cells, faces, nodes, boundaries = build_mesh(
    "unv_sample_meshes/quad.unv", scale=0.01)

scatter(nodes, aspect_ratio=:equal)

fig = plot(aspect_ratio=:equal, color=:blue, legend=false)
for fID ∈ 1:length(faces)
    p1 = nodes[faces[fID].nodesID[1]].coords
    p2 = nodes[faces[fID].nodesID[2]].coords
    x = [p1[1], p2[1]]
    y = [p1[2], p2[2]]
    plot!(fig, x,y, color=:blue)
end
@show fig