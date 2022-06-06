using Plots
using LinearOperators
using LinearAlgebra

using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.Discretise
using FVM_1D.Calculate
using FVM_1D.Models
using FVM_1D.Solvers
using FVM_1D.VTK

using Krylov

function generate_mesh()
    # n_vertical      = 400 #20 #400
    # n_horizontal1   = 500 #25 #500
    # n_horizontal2   = 400 #20 #800

    n_vertical      = 20 #400
    n_horizontal1   = 25 #500
    n_horizontal2   = 20 #800

    p1 = Point(0.0,0.0,0.0)
    p2 = Point(1.0,0.2,0.0)
    # p2 = Point(1.0,0.0,0.0)
    p3 = Point(1.5,0.2,0.0)
    # p3 = Point(1.5,0.0,0.0)
    p4 = Point(0.0,1.0,0.0)
    p5 = Point(1.0,0.8,0.0)
    # p5 = Point(1.0,1.0,0.0)
    p6 = Point(1.5,0.8,0.0)
    # p6 = Point(1.5,0.7,0.0)
    points = [p1, p2, p3, p4, p5, p6]

    # Edges in x-direction
    e1 = line!(points,1,2,n_horizontal1)
    e2 = line!(points,2,3,n_horizontal2)
    e3 = line!(points,4,5,n_horizontal1)
    e4 = line!(points,5,6,n_horizontal2)

    # Edges in y-direction
    e5 = line!(points,1,4,n_vertical)
    e6 = line!(points,2,5,n_vertical)
    e7 = line!(points,3,6,n_vertical)
    edges = [e1, e2, e3, e4, e5, e6, e7]

    b1 = quad(edges, [1,3,5,6])
    b2 = quad(edges, [2,4,6,7])
    blocks = [b1, b2]

    patch1 = Patch(:inlet,  [5])
    patch2 = Patch(:outlet, [7])
    patch3 = Patch(:bottom, [1,2])
    patch4 = Patch(:top,    [3,4])
    patches = [patch1, patch2, patch3, patch4]

    builder = MeshBuilder2D(points, edges, patches, blocks)
    mesh = generate!(builder)
    return mesh
end

function create_model(::Type{ConvectionDiffusion}, U, J, phi)
    model = ConvectionDiffusion(
        Divergence{Linear}(U, phi),
        Laplacian{Linear}(J, phi),
        0.0
        )
    model.terms.term2.sign[1] = -1
    return model
end

function create_model(::Type{Diffusion}, J, phi)
    model = Diffusion(
        Laplacian{Linear}(J, phi),
        0.0
        )
    return model
end

BCs = (
    Dirichlet(:inlet, 100.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
    # Dirichlet(:bottom, 50.0),
    # Dirichlet(:top, 50.0)
)

UBCs = (
    Dirichlet(:inlet, [1.0, 0.0, 0.0]),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
    # Dirichlet(:bottom, 50.0),
    # Dirichlet(:top, 50.0)
)

setup = SolverSetup(
    iterations  = 100,
    solver      = GmresSolver,
    tolerance   = 1e-6,
    relax       = 1.0,
    itmax       = 100
    )

using JLD2

mesh = generate_mesh()
phi = ScalarField(mesh)

jldopen("data/mesh.jld2", "w") do file
    file["mesh"] = mesh
end
mesh = load("data/mesh.jld2", "mesh")

jldopen("data/fields.jld2", "w") do file
    file["phi"] = phi
end
fields = load("data/fields.jld2")
fields["phi"]

equation = Equation(mesh)
phiModel = create_model(ConvectionDiffusion, [4.0, 0.0, 0.0], 1.0, phi)
# phiModel = create_model(Diffusion, 1.0, phi)
generate_boundary_conditions!(mesh, phiModel, BCs)

discretise!(equation, phiModel)
update_boundaries!(equation, phiModel, BCs)

clear!(phi)
@time run!(equation, phiModel, BCs, setup)
write_vtk(mesh, phi)

divU = ScalarField(mesh)
U = VectorField(mesh) #### HEREE!!!
Uf = FaceVectorField(mesh)
Divergence{Linear}(Uf, phi)
@time div!(divU, Uf, U, BCs) # input here should an an object of type Div

@time initialise!(Uf, [21.0, 5, 0])
Uf(30)
phiModel = create_model(ConvectionDiffusion, Uf, 1.0, phi)


(; A, b, R, Fx) = equation
@time Diagonal(A)
@time @view A[diagind(A)]
@time Diagonal(A[diagind(A)])
@time Diagonal(@view A[diagind(A)])
@time C = A .- Diagonal(@view A[diagind(A)])
@time S = sum(C*I, dims=2)

### Non-orthogonal correction

phi1 = ScalarField(mesh)

jldopen("data/fields.jld2", "w") do file
    file["phi1"] = phi1
end
fields = load("data/fields.jld2")
fields["phi1"]

phiModel = create_model(ConvectionDiffusion, [4.0, 0.0, 0.0], 1.0, phi1)
# phiModel = create_model(Diffusion, 1.0, phi1)
generate_boundary_conditions!(mesh, phiModel, BCs)

setup = SolverSetup(
    iterations  = 100,
    solver      = GmresSolver,
    tolerance   = 1e-6,
    relax       = 0.6,
    itmax       = 100
    )

GC.gc()
term = phiModel.terms.term2 # For convection-diffusion model
# term = phiModel.terms.term1 # For pure diffusion
clear!(phi1)
@time run!(equation, phiModel, BCs, setup, correct_term=term)


# gr(size=(400,400), camera=(45,55))
plotly(size=(400,400), markersize=1, markerstrokewidth=1)
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

# for (nodei, node) ∈ enumerate(mesh.nodes)
#     centre = [node.coords[1]], [node.coords[2]]
#     fig = annotate!(fig, centre[1],centre[2], text("$(nodei)", :black, 8))
# end
# for (facei, face) ∈ enumerate(mesh.faces)
#     centre = centre2d(face)
#     fig = annotate!(fig, centre[1],centre[2], text("$(facei)", :blue, 10))
# end

for (celli, cell) ∈ enumerate(mesh.cells)
    centre = centre2d(cell)
    fig = annotate!(fig, centre[1],centre[2], text("$(celli)", :red, 10))
end
@show fig
