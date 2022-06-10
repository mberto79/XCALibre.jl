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

velocity = [0.1, 0.0, 0.0]
nu = 0.1
Re = velocity[1]*0.5/nu

UBCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

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
    relax       = 0.3,
    itmax       = 100,
    rtol        = 1e-2
)

mesh = generate_mesh()
U = VectorField(mesh)
Uf = FaceVectorField(mesh)
Hv = VectorField(mesh)
divHv = Div(Hv)
ux = ScalarField(mesh)
uy = ScalarField(mesh)
rD = ScalarField(mesh)
rDf = FaceScalarField(mesh)

p = ScalarField(mesh)
pf = FaceScalarField(mesh)
∇p = Grad{Linear}(p)

x_momentum_eqn = Equation(mesh)
x_momentum_model = create_model(ConvectionDiffusion, Uf, nu, ux, ∇p.x)

y_momentum_eqn = Equation(mesh)
y_momentum_model = create_model(ConvectionDiffusion, Uf, nu, uy, ∇p.y)

pressure_eqn = Equation(mesh)
pressure_correction = create_model(Diffusion, rDf, p, divHv.values) #.*D)

set!(p, x, y) = begin
    # inlet_value = 0.0 #2
    # p.values .= inlet_value .- inlet_value*x
    p.values .= 0.0
end

set!(p, x(mesh), y(mesh))
U.x .= velocity[1]; U.y .= velocity[2]
interpolate!(Uf, U, UBCs)

clear!(ux)
clear!(uy)
clear!(p)

# ux0 = zeros(length(ux.values))
# uy0 = zeros(length(ux.values))
p0 = zeros(length(p.values))
cvols = volumes(mesh)
for i ∈ 1:10

source!(∇p, pf, p, pBCs)
# grad!(∇p, pf, p, pBCs)
∇p.x .*= -1.0
∇p.y .*= -1.0

discretise!(x_momentum_eqn, x_momentum_model)
generate_boundary_conditions!(mesh, x_momentum_model, uxBCs)
update_boundaries!(x_momentum_eqn, x_momentum_model, uxBCs)
@time run!(x_momentum_eqn, x_momentum_model, uxBCs, setup)
write_vtk(mesh, ux)

discretise!(y_momentum_eqn, y_momentum_model)
generate_boundary_conditions!(mesh, y_momentum_model, uyBCs)
update_boundaries!(y_momentum_eqn, y_momentum_model, uyBCs)
@time run!(y_momentum_eqn, y_momentum_model, uyBCs, setup)
write_vtk(mesh, uy)

# α = 0.2
# U.x .= α*ux.values + (1.0 - α)*U.x
# U.y .= α*uy.values + (1.0 - α)*U.y# make U.x a reference to ux.values etc.

U.x .= ux.values
U.y .= uy.values


D = @view x_momentum_eqn.A[diagind(x_momentum_eqn.A)]
rD.values .= 1.0./D
interpolate!(rDf, rD)
@time H!(Hv, U, x_momentum_eqn, y_momentum_eqn)
@time div!(divHv, UBCs) 
# divHv.values .*= cvols

discretise!(pressure_eqn, pressure_correction)
generate_boundary_conditions!(mesh, pressure_correction, pBCs)
update_boundaries!(pressure_eqn, pressure_correction, pBCs)
@time run!(pressure_eqn, pressure_correction, pBCs, setup)
write_vtk(mesh, p)

β = 0.2
p.values .= β*p.values + (1.0 - β)*p0
p0 .= p.values

grad!(∇p, pf, p, pBCs) # something off with gradient calculation!
# ux.values .= Hv.x .- ∇p.x.*rD.values
# uy.values .= Hv.y .- ∇p.y.*rD.values #./(D)

U.x .= Hv.x .- ∇p.x.*rD.values
U.y .= Hv.y .- ∇p.y.*rD.values #./(D)
interpolate!(Uf, U, UBCs)

# U.x .= ux.values
# U.y .= uy.values

# ux.values .= U.x
# uy.values .= U.y

# clear!(ux)
# clear!(uy)


end
# ux.values .= U.x
# uy.values .= U.y
write_vtk(mesh, ux)
write_vtk(mesh, uy)


plotly(size=(400,400), markersize=1, markerstrokewidth=1)
scatter(x(mesh), y(mesh), ux.values, color=:red)
scatter(x(mesh), y(mesh), U.x, color=:green)
scatter(x(mesh), y(mesh), U.y, color=:green)
scatter(x(mesh), y(mesh), divHv.values, color=:red)
scatter(x(mesh), y(mesh), rD.values, color=:red)
scatter(x(mesh), y(mesh), Hv.x./D, color=:green)
scatter(x(mesh), y(mesh), ∇p.x, color=:green)
scatter!(x(mesh), y(mesh), p.values, color=:blue)
scatter(xf(mesh), yf(mesh), divHv.face_vector.x, color=:red)
scatter(xf(mesh), yf(mesh), Uf.x, color=:red)
scatter(xf(mesh), yf(mesh), Uf.y, color=:red)
scatter(xf(mesh), yf(mesh), pf.values, color=:red)
scatter(xf(mesh), yf(mesh), rDf.values, color=:red)


function volumes(mesh)
    (; cells) = mesh
    vols = Float64[]
    for cell ∈ cells 
        push!(vols, cell.volume)
    end
    vols
end