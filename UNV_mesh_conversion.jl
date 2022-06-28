using Plots

using FVM_1D.Mesh2D
using FVM_1D.UNV

points, elements, boundaries = load("unv_sample_meshes/quad.unv", Int64, Float64)

build_mesh("unv_sample_meshes/quad.unv")

fig = scatter()
for point ∈ points 
    xyz = point.xyz 
    x = xyz[1]
    y = xyz[2]
    z = xyz[3]
    scatter!(fig, [x],[y], color=:blue, legend=false)
end

@show fig