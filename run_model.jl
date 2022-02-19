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
GC.gc()
# f = plot(mesh.nodes, :coords; labels=true)
# plot!(f, mesh.cells, :centre; labels=true)

@time temp =
 Δ{Linear}(k,ϕ) + Δ{Linear}(0.0,ϕ)
temp.A.nzval
ϕ.equation.A.nzval
@time apply_boundary_conditions!(ϕ, k, 300, 100)
@time FVM_1D.solve!(ϕ)
ϕ.values
clear!(ϕ.equation)
clear!(ϕ)

@time modelEquation = Δ{Linear}(k, ϕ) == Source{Constant}(0.0)
discretise! = @discretise modelEquation

term = Δ{Linear}(k, ϕ)
@time discretise!(ϕEqn, term)

clear!(ϕ.equation)
ϕ.values

x(mesh) = [mesh.cells[i].centre[1] for i ∈ 1:length(mesh.cells)]
Plots.scatter(x(mesh), ϕ.values)