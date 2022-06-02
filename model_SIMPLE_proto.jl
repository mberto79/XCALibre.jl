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

    n_vertical      = 40 #400
    n_horizontal    = 80 #500

    p1 = Point(0.0,0.0,0.0)
    p2 = Point(1.0,0.0,0.0)
    p3 = Point(0.0,0.5,0.0)
    p4 = Point(1.0,0.5,0.0)
    points = [p1, p2, p3, p4]

    # Edges in x-direction
    e1 = line!(points,1,2,n_horizontal)
    e2 = line!(points,3,4,n_horizontal)
    
    # Edges in y-direction
    e3 = line!(points,1,3,n_vertical)
    e4 = line!(points,2,4,n_vertical)
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

function create_model(::Type{ConvectionDiffusion}, U, J, phi, S)
    model = ConvectionDiffusion(
        Divergence{Linear}(U, phi),
        Laplacian{Linear}(J, phi),
        S
        )
    model.terms.term2.sign[1] = -1
    return model
end

function create_model(::Type{Diffusion}, J, phi, S)
    model = Diffusion(
        Laplacian{Linear}(J, phi),
        S
        )
    return model
end

UBCs = ( 
    Dirichlet(:inlet, [0.1, 0.0, 0.0]),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

velocity = [0.2, 0.0, 0.0]
nu = 1.5e-1

uxBCs = (
    Dirichlet(:inlet, velocity[1]),
    Neumann(:outlet, 0.0),
    # Dirichlet(:outlet, 0.0),
    Dirichlet(:bottom, 0.0),
    Dirichlet(:top, 0.0)
)

uyBCs = (
    Dirichlet(:inlet, velocity[2]),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, 0.0),
    Dirichlet(:top, 0.0)
)

pBCs = (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

setup = SolverSetup(
    iterations  = 100,
    solver      = GmresSolver,
    tolerance   = 1e-6,
    relax       = 1.0,
    itmax       = 100
)

mesh = generate_mesh()
U = VectorField(mesh)
divU = ScalarField(mesh)
Uf = FaceVectorField(mesh); initialise!(Uf, velocity)
ux = ScalarField(mesh)
uy = ScalarField(mesh)

p = ScalarField(mesh)
set!(p, x, y) = begin
    inlet_value = 2
    p.values .= inlet_value .- inlet_value*x
end
set!(p, x(mesh), y(mesh))
pf = FaceScalarField(mesh)
∇p = Grad{Linear}(p)
source!(∇p, pf, p, pBCs)

momentum_eqn = Equation(mesh)
momentum_x = create_model(ConvectionDiffusion, Uf, nu, ux, -∇p.x)
generate_boundary_conditions!(mesh, momentum_x, uxBCs)
clear!(ux)
@time run!(momentum_eqn, momentum_x, uxBCs, setup)
write_vtk(mesh, ux)

momentum_eqn = Equation(mesh)
momentum_y = create_model(ConvectionDiffusion, Uf, nu, uy, -∇p.y)
generate_boundary_conditions!(mesh, momentum_y, uyBCs)
clear!(uy)
@time run!(momentum_eqn, momentum_y, uyBCs, setup)
write_vtk(mesh, uy)

U.x .= ux.values
U.y .= uy.values
@time div!(divU, Uf, U, UBCs) 

D = @view A[diagind(A)]
rD = ScalarField(mesh)
rD.values .= 1.0./D
pressure_correction = create_model(Diffusion, rD, p, divU.values.*rD)


(; A, b, R, Fx) = momentum_eqn
@time Diagonal(A)
@time D = @view A[diagind(A)]
@time Diagonal(A[diagind(A)])
@time Diagonal(@view A[diagind(A)])
@time C = A .- Diagonal(@view A[diagind(A)])
@time S = sum(C*I, dims=2)

plotly(size=(400,400), markersize=1, markerstrokewidth=1)
scatter(x(mesh), y(mesh), ux.values, color=:red)
scatter(x(mesh), y(mesh), uy.values, color=:red)
scatter(x(mesh), y(mesh), divU.values, color=:red)
scatter(x(mesh), y(mesh), p.values, color=:red)
scatter(xf(mesh), yf(mesh), pf.values, color=:red)
scatter(x(mesh), y(mesh), ∇p.x, color=:green)
