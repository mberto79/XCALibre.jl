using Plots
using SparseArrays
using Krylov

using FVM_1D
# plotly()

nCells = Int(10)
x0 = 0.0
xL = 2.0
h = 0.5
k = 5.0

GC.gc()
mesh = generate_mesh_1D(x0, xL, h, nCells)
ϕ = ScalarField(zeros(length(mesh.cells)), mesh)
ϕEqn = Equation(ϕ)

# f = plot(mesh.nodes, :coords; labels=true)
# plot!(f, mesh.cells, :centre; labels=true)

@time modelEquation = Δ{Linear}(k, ϕ) == Source{Constant}(0.0)
discretise! = @discretise modelEquation

term = Δ{Linear}(k, ϕ)
@time discretise!(ϕEqn, term)
@time apply_boundary_conditions!(ϕEqn, k, 300, 100)
@time FVM_1D.solve!(ϕEqn)

clear!(ϕEqn)

x(mesh) = [mesh.cells[i].centre[1] for i ∈ 1:length(mesh.cells)]
Plots.scatter(x(mesh), ϕ.values)