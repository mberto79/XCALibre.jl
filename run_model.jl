using Plots

using FVM_1D
# plotly()

nCells = Int(50)
x0 = 0.0
xL = 2.0
h = 0.5
k = 5.0  
U = [1.5, 0.1, 0.0]
# U = [0.0, 0.0, 0.0]

@time mesh = generate_mesh_1D(x0, xL, h, nCells)
GC.gc()

@time equation = Equation(mesh)
@time ϕ = ScalarField(mesh, equation)
# f = plot(mesh.nodes, :coords; labels=true)
# plot!(f, mesh.cells, :centre; labels=true)

term1 = Laplacian{Linear}(k, ϕ)
source1 = 0.0
@time ϕModel = SteadyDiffusion(term1, source1)
@time discretise!(ϕModel, equation)
@time apply_boundary_conditions!(equation, mesh, k, 300, 100)
@time solve!(ϕ)
ϕ.values
function run!(ϕModel, mesh, equation, k, U)
    term1 = Divergence{Linear}(U, ϕ)
    term2 = Laplacian{Linear}(k, ϕ)
    term2.sign[1] = -1
    source1 = 0.0
    @time ϕModel = SteadyConvectionDiffusion(term1,term2, source1)
    @time discretise!(ϕModel, equation)
    @time apply_boundary_conditions!(equation, mesh, k, U, 300.0, 0.0)
    @time solve!(ϕ)
    ϕ.values
    nothing
end

run!(ϕModel, mesh, equation, k, U)

x(mesh) = [mesh.cells[i].centre[1] for i ∈ 1:length(mesh.cells)]
Plots.scatter(x(mesh), ϕ.values)