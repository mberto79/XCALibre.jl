using Plots
using SparseArrays
using Krylov

using FVM_1D
# revise(FVM_1D)
# plotly()

nCells = Int(10)
x0 = 0.0
xL = 2.0
h = 0.5
k = 5.0

GC.gc()
@time mesh = generate_mesh_1D(x0, xL, h, nCells)
@time ϕ = ScalarField(mesh)
GC.gc()
# f = plot(mesh.nodes, :coords; labels=true)
# plot!(f, mesh.cells, :centre; labels=true)

term1 = Laplacian{Linear}(k, ϕ)
source1 = 0.0
@time ϕModel = SteadyDiffusion(term1, source1)

@time generalDiscretise!(ϕModel, aP!{Linear}, aN!{Linear}, b!{Linear})
@code_warntype generalDiscretise!(ϕModel, aP!{Linear}, aN!{Linear}, b!{Linear})
@time apply_boundary_conditions!(ϕ, k, 300, 100)
ϕModel.equation.A
ϕModel.equation.b

FVM_1D.solve!(ϕ)
ϕ.values

clear!(ϕModel.equation)

x(mesh) = [mesh.cells[i].centre[1] for i ∈ 1:length(mesh.cells)]
Plots.scatter(x(mesh), ϕ.values)