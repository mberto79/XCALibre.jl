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
Uf = FaceVectorField(mesh)
Uf.x .= velocity[1]; Uf.y .= velocity[2]
Hv = VectorField(mesh)
divHv = Div(Hv)
ux = ScalarField(mesh)
uy = ScalarField(mesh)

p = ScalarField(mesh)
set!(p, x, y) = begin
    inlet_value = 0.0 #2
    p.values .= inlet_value .- inlet_value*x
end
set!(p, x(mesh), y(mesh))
pf = FaceScalarField(mesh)
∇p = Grad{Linear}(p)


source!(∇p, pf, p, pBCs)

x_momentum_eqn = Equation(mesh)
x_momentum_model = create_model(ConvectionDiffusion, Uf, nu, ux, -∇p.x)
generate_boundary_conditions!(mesh, x_momentum_model, uxBCs)
discretise!(x_momentum_eqn, x_momentum_model)
update_boundaries!(x_momentum_eqn, x_momentum_model, uxBCs)
clear!(ux)
@time run!(x_momentum_eqn, x_momentum_model, uxBCs, setup)
write_vtk(mesh, ux)

y_momentum_eqn = Equation(mesh)
y_momentum_model = create_model(ConvectionDiffusion, Uf, nu, uy, -∇p.y)
generate_boundary_conditions!(mesh, y_momentum_model, uyBCs)
discretise!(y_momentum_eqn, y_momentum_model)
update_boundaries!(y_momentum_eqn, y_momentum_model, uyBCs)
clear!(uy)
@time run!(y_momentum_eqn, y_momentum_model, uyBCs, setup)
write_vtk(mesh, uy)

U.x .= ux.values; U.y .= uy.values # make U.x a reference to ux.values etc.

D = @view x_momentum_eqn.A[diagind(x_momentum_eqn.A)]
@time H!(Hv, U, x_momentum_eqn, y_momentum_eqn)
@time div!(divHv, UBCs) 

pressure_eqn = Equation(mesh)
pressure_correction = create_model(Diffusion, 1.0, p, divHv.values.*D)
generate_boundary_conditions!(mesh, pressure_correction, pBCs)
discretise!(pressure_eqn, pressure_correction)
update_boundaries!(pressure_eqn, pressure_correction, pBCs)
clear!(p)
@time run!(pressure_eqn, pressure_correction, pBCs, setup)
write_vtk(mesh, p)

grad!(∇p, pf, p, pBCs)
U.x .= Hv.x .- ∇p.x./(D)
U.y .= Hv.y .- ∇p.y./(D)

ux.values .= U.x
uy.values .= U.y
write_vtk(mesh, ux)
write_vtk(mesh, uy)

interpolate!(Uf, U, UBCs)



plotly(size=(400,400), markersize=1, markerstrokewidth=1)
scatter(x(mesh), y(mesh), ux.values, color=:red)
scatter!(x(mesh), y(mesh), U.y, color=:green)
scatter(x(mesh), y(mesh), divHv.values, color=:red)
scatter(x(mesh), y(mesh), Hv.x, color=:green)
scatter(x(mesh), y(mesh), p.values, color=:red)
scatter(xf(mesh), yf(mesh), divHv.face_vector.x, color=:red)
scatter(x(mesh), y(mesh), ∇p.x, color=:green)
