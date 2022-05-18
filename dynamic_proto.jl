using Plots
using LinearOperators
using LinearAlgebra

using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.Discretise
using FVM_1D.Calculate
using FVM_1D.Models
using FVM_1D.Solvers

using Krylov

function generate_mesh()
    n_vertical      = 400 #20 #400
    n_horizontal1   = 500 #25 #500
    n_horizontal2   = 400 #20 #800

    # n_vertical      = 20 #400
    # n_horizontal1   = 25 #500
    # n_horizontal2   = 20 #800

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

mesh = generate_mesh()
phi = ScalarField(mesh)
equation = Equation(mesh)
phiModel = create_model(ConvectionDiffusion, [4.0, 0.0, 0.0], 1.0, phi)
phiModel = create_model(Diffusion, 1.0, phi)
generate_boundary_conditions!(mesh, phiModel, BCs)

@time discretise!(equation, phiModel, mesh)
@time update_boundaries!(equation, mesh, phiModel, BCs)

# @time system = set_solver(equation, GmresSolver)
system = set_solver(equation, BicgstabSolver)

@time run!(system, equation, phi; itmax=2000)#, history=true)
# update_residual!(opA, equation, phi)
update_residual!(opA, equation, phi)
residual(equation)

using IncompleteLU
(; A) = equation
(; m, n) = A
F = ilu(A, τ = 0.005)

# Definition of linear operators to reduce allocations during iterations
opP = LinearOperator(Float64, A.m, A.n, false, false, (y, v) -> ldiv!(y, F, v))
opA = LinearOperator(A)
phi.values .= 0.0
system.x .= 0.0
α = 0.4
@time for i ∈ 1:500

    # Solving in residual form (allowing to provide an initial guess)
    # update_residual!(opA, equation, phi)
    # solve!(system, opA, equation.R; M=opP, itmax=100, atol=1e-12, rtol=1e-3)
    solve!(system, opA, equation.b, system.x; M=opP, itmax=10, atol=1e-12, rtol=1e-3)
    phi.values .= (1.0 - α).*phi.values .+ α.*system.x
    # update_solution!(phi, system; alpha=1.0) # adds solution to initial guess

    update_residual!(opA, equation, phi)
    if residual(equation) <= 1e-6
        # update_residual!(opA, equation, phi)
        residual_print(equation)
        println("Converged in ", i, " iterations")
        break
    end
end
phi.values .= system.x # A: 30.42s (1.79k), opA = 30.39s (1.39k), opA(free) = 32.59 (969)
update_residual!(opA, equation, phi)
residual(equation)

phi1 = ScalarField(mesh)
phiModel = create_model(Diffusion, 1.0, phi1)
generate_boundary_conditions!(mesh, phiModel, BCs)

phif = FaceScalarField(mesh)
gradf = FaceVectorField(mesh)

gradPhi = Grad{Linear}(phi1)
gradPhi = Grad{Linear}(phi1,2)

system = set_solver(equation, BicgstabSolver)
discretise!(equation, phiModel, mesh)
update_boundaries!(equation, mesh, phiModel, BCs)
F = ilu(equation.A, τ = 0.005)
# Definition of linear operators to reduce allocations during iterations
m = equation.A.m; n = m
opP = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, F, v))
opA = LinearOperator(equation.A)
phi1.values .= 0.0
# update_residual!(opA, equation, phi1)
discretise!(equation, phiModel, mesh)
update_boundaries!(equation, mesh, phiModel, BCs)
update_residual!(opA, equation, phi1)
system.x .= 0.0
phi1.values .= 0.0
α = 0.6
@time for i ∈ 1:100 #50
    # Solving in residual form (allowing to provide an initial guess)
    # solve!(system, opA, equation.R; M=opP, itmax=5, atol=1e-12, rtol=1e-3)
    # system.x .= phi1.values
    
    # warm_start!(system, phi1.values)
    solve!(system, opA, equation.b, phi1.values; M=opP, itmax=20, atol=1e-12, rtol=1e-3)
    phi1.values .= (1.0 - α).*phi1.values .+ α.*system.x
    
    # update_solution!(phi1, system) # adds solution to initial guess
    # update_residual!(opA, equation, phi1)
    
    nonorthogonal_correction!(gradPhi, gradf, phif, BCs)

    discretise!(equation, phiModel, mesh)
    update_boundaries!(equation, mesh, phiModel, BCs)
    term = phiModel.terms.term1
    correct!(equation, term, phif)

    update_residual!(opA, equation, phi1)
    if residual(equation) <= 1e-6
        residual_print(equation)
        println("Converged in ", i, " iterations")
        break
    end
end
update_residual!(opA, equation, phi1)
residual(equation)
phi1.values .= system.x

# gr(size=(400,400), camera=(45,55))
plotly(size=(400,400), markersize=1, markerstrokewidth=1)
f(x,y) = 100 + (0.0 - 100)/(1.5 - 0.0)*x
surface(x(mesh), y(mesh), f)
scatter!(x(mesh), y(mesh), phi.values, color=:blue)
scatter!(x(mesh), y(mesh), phi1.values, color=:green)
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
