using Plots
using SparseArrays
using Krylov

using FVM_1D
revise(FVM_1D)
# plotly()

nCells = Int(10)
x0 = 0.0
xL = 2.0
h = 0.5
k = 5.0

GC.gc()
@time mesh = generate_mesh_1D(x0, xL, h, nCells)
@time ϕ = ScalarField(mesh)
isbits(ϕ)
GC.gc()
# f = plot(mesh.nodes, :coords; labels=true)
# plot!(f, mesh.cells, :centre; labels=true)

@time ϕModel = SteadyDiffusion{Linear}(Laplacian{Linear}(k, ϕ), [1], ϕ.equation)
isbits(ϕModel)
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