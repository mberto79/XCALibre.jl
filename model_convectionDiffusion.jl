using Plots
using LinearOperators
using LinearAlgebra
using Statistics

using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.Discretise
using FVM_1D.Calculate
using FVM_1D.Models
using FVM_1D.Solvers
using FVM_1D.VTK

using Krylov
using ILUZero
using IncompleteLU
using LoopVectorization

function generate_mesh(n_horizontal, n_vertical)

    p1 = Point(0.0,0.0,0.0)
    p2 = Point(1.0,0.0,0.0)
    # p2 = Point(0.5,-0.2,0.0)
    p3 = Point(0.0,1.0,0.0)
    p4 = Point(1.0,1.0,0.0)
    points = [p1, p2, p3, p4]

    # Edges in x-direction
    e1 = line!(points,1,2,n_horizontal)
    e2 = line!(points,3,4,n_horizontal)
    
    # Edges in y-direction
    e3 = line!(points,1,3,n_vertical)
    e4 = line!(points,2,4,n_vertical)
    # e3 = line!(points,1,3,n_vertical,4) # with stretching
    # e4 = line!(points,2,4,n_vertical,4) # with stretching
    edges = [e1, e2, e3, e4]

    b1 = quad(edges, [1,2,3,4])
    blocks = [b1]

    patch1 = Patch(:inlet,  [3])
    patch2 = Patch(:outlet, [4])
    patch3 = Patch(:bottom, [1])
    patch4 = Patch(:top,    [2])
    patches = [patch1, patch2, patch3, patch4]

    builder = MeshBuilder2D(points, edges, patches, blocks)
    mesh = generate!(builder)
    return mesh
end

BCs = (
    Dirichlet(:inlet, 100.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
    # Dirichlet(:bottom, 50.0),
    # Dirichlet(:top, 50.0)
)

setup = SolverSetup(
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-2
)

mesh = generate_mesh(20, 20)
equation = Equation(mesh)

velocity = [10, 0.0, 0.0]
gamma = 1.0

phi = ScalarField(mesh)
Uf = FaceVectorField(mesh); Uf.x .= velocity[1]; Uf.y .= velocity[2]; Uf.z .= velocity[3]
mdotf = FaceScalarField(mesh)
flux!(mdotf, Uf)
source = zeros(Float64, length(mesh.cells))

phiModel = create_model(ConvectionDiffusion, mdotf, gamma, phi, source)


discretise!(equation, phiModel)
apply_boundary_conditions!(equation, phiModel, BCs)

phi = equation.A\equation.b


plotly(size=(400,400), markersize=1, markerstrokewidth=1)
scatter(x(mesh), y(mesh), phi)





# OLD STUFF




f(x,y) = 100 + (0.0 - 100)/(1.5 - 0.0)*x
surface(x(mesh), y(mesh), f)
scatter(x(mesh), y(mesh), phi.values, color=:blue)
scatter!(x(mesh), y(mesh), phi1.values, color=:red)
scatter(xf(mesh), yf(mesh), phif.values, color=:green)
scatter(x(mesh), y(mesh), gradPhi.x, color=:blue)
scatter(x(mesh), y(mesh), gradPhi.x, color=:green)
scatter!(xf(mesh), yf(mesh), gradf.x, color=:red)


scatter(mesh.nodes, colour=:black)
scatter!(centre2d.(mesh.faces), color=:blue)
scatter!(centre2d.(mesh.cells), color=:red)
plot_mesh!(mesh)
fig = plot_mesh!(mesh)

for boundary ∈ mesh.boundaries
    for ID ∈ boundary.facesID
        face = mesh.faces[ID]
        normal = 0.1*face.normal
        centre = centre2d(face)
        fig = quiver!(fig, centre..., quiver=([normal[1]], [normal[2]]), color=:green)
    end
end
@show fig
for face ∈ mesh.faces
    normal = 0.1*face.normal
    centre = centre2d(face)
    fig = quiver!(fig, centre..., quiver=([normal[1]], [normal[2]]), color=:green)
end
@show fig

for (celli, cell) ∈ enumerate(mesh.cells)
    centre = centre2d(cell)
    fig = annotate!(fig, centre[1],centre[2], text("$(celli)", :red, 10))
end
@show fig

jldopen("data/mesh.jld2", "w") do file
    file["mesh"] = mesh
end
mesh = load("data/mesh.jld2", "mesh")

jldopen("data/fields.jld2", "w") do file
    file["phi"] = phi
end
fields = load("data/fields.jld2")
fields["phi"]
