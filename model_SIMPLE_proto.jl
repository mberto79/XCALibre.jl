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
    # n_vertical      = 20 
    # n_horizontal    = 100 

    n_vertical      = 40 
    n_horizontal    = 200 

    p1 = Point(0.0,0.0,0.0)
    p2 = Point(0.5,0.0,0.0)
    p3 = Point(0.0,0.1,0.0)
    p4 = Point(0.5,0.1,0.0)
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

velocity = [0.5, 0.0, 0.0]
nu = 0.001
Re = velocity[1]*0.1/nu

UBCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
    # Neumann(:top, 0.0)
)

uxBCs = (
    Dirichlet(:inlet, velocity[1]),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

uyBCs = (
    Dirichlet(:inlet, velocity[2]),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

pBCs = (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

setup = SolverSetup(
    iterations  = 100,
    solver      = BicgstabSolver,
    tolerance   = 1e-6,
    # tolerance   = 1e-01,
    relax       = 1.0,
    itmax       = 100,
    rtol        = 1e-4
)

setup_p = SolverSetup(
    iterations  = 100,
    solver      = BicgstabSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    tolerance   = 1e-6,
    # tolerance   = 1e-01,
    relax       = 1.0,
    itmax       = 100,
    rtol        = 1e-4
)

#SymmlqSolver, MinresSolver - did not work!

mesh = generate_mesh()

ux = ScalarField(mesh)
uy = ScalarField(mesh)

U = VectorField(mesh)
# U = VectorField(ux.values, uy.values, zeros(eltype(ux.values), length(ux.values)), mesh)
Uf = FaceVectorField(mesh)



Hv = VectorField(mesh)
divHv = Div(Hv)

rD = ScalarField(mesh)
rDf = FaceScalarField(mesh)

p = ScalarField(mesh)
pf = FaceScalarField(mesh)
∇p = Grad{Linear}(p)

x_momentum_eqn = Equation(mesh)
x_momentum_model = create_model(ConvectionDiffusion, Uf, nu, ux, ∇p.x)
generate_boundary_conditions!(:ux_boundary_update!, mesh, x_momentum_model, uxBCs)

y_momentum_eqn = Equation(mesh)
y_momentum_model = create_model(ConvectionDiffusion, Uf, nu, uy, ∇p.y)
generate_boundary_conditions!(:uy_boundary_update!, mesh, y_momentum_model, uyBCs)

pressure_eqn = Equation(mesh)
pressure_correction = create_model(Diffusion, rDf, p, divHv.values) #.*D)
generate_boundary_conditions!(:p_boundary_update!, mesh, pressure_correction, pBCs)

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

ux0 = zeros(length(ux.values))
ux0 .= velocity[1]
uy0 = zeros(length(ux.values))
uy0 .= velocity[2]
p0 = zeros(length(p.values))
B = zeros(length(mesh.cells),3)
V = zeros(length(mesh.cells),3)
H = zeros(length(mesh.cells),3)

############################
#############################

@time for i ∈ 1:200

println("Iteration ", i)

source!(∇p, pf, p, pBCs)
negative_vector_source!(∇p)
# ∇p.x .*= -1.0
# ∇p.y .*= -1.0

discretise!(x_momentum_eqn, x_momentum_model)
Discretise.ux_boundary_update!(x_momentum_eqn, x_momentum_model, uxBCs)
println("Solving x-momentum")
run!(x_momentum_eqn, x_momentum_model, uxBCs, setup)

discretise!(y_momentum_eqn, y_momentum_model)
Discretise.uy_boundary_update!(y_momentum_eqn, y_momentum_model, uyBCs)
println("Solving y-momentum")
run!(y_momentum_eqn, y_momentum_model, uyBCs, setup)

inverse_diagonal!(rD, x_momentum_eqn)
interpolate!(rDf, rD)
remove_pressure_source!(x_momentum_eqn, y_momentum_eqn)
@. U.x = ux.values 
@. U.y = uy.values
H!(Hv, U, x_momentum_eqn, y_momentum_eqn, B, V, H)
div!(divHv, UBCs) 

discretise!(pressure_eqn, pressure_correction)
Discretise.p_boundary_update!(pressure_eqn, pressure_correction, pBCs)
println("Solving pressure correction")
run!(pressure_eqn, pressure_correction, pBCs, setup_p, precondition=true)

explicit_relaxation!(p, p0, 0.3)

source!(∇p, pf, p, pBCs) 

correct_velocity!(U, ∇p, rD)
interpolate!(Uf, U, UBCs)
@. ux.values = U.x
@. uy.values = U.y

end # 4.6s, 4.4s
write_vtk(mesh, ux)
write_vtk(mesh, uy)
write_vtk(mesh, p)

function inverse_diagonal!(rD::ScalarField, eqn)
    D = @view eqn.A[diagind(eqn.A)]
    rD.values .= 1.0./D
    nothing
end

function explicit_relaxation!(phi, phi0, alpha)
    @. phi.values = alpha*phi.values + (1.0 - alpha)*phi0
    @. phi0 = phi.values
    nothing
end

function correct_velocity!(U, ∇p, rD)
    @. U.x = Hv.x - ∇p.x #*rD.values
    @. U.y = Hv.y - ∇p.y #*rD.values
    nothing
end

function negative_vector_source!(∇p)
    ∇p.x .*= -1.0
    ∇p.y .*= -1.0
    nothing
end

function remove_pressure_source!(x_momentum_eqn, y_momentum_eqn)
    @. x_momentum_eqn.b -= ∇p.x
    @. y_momentum_eqn.b -= ∇p.y
    nothing
end

plotly(size=(400,400), markersize=1, markerstrokewidth=1)
scatter(x(mesh), y(mesh), ux.values, color=:red)
scatter(x(mesh), y(mesh), uy.values, color=:red)

scatter(x(mesh), y(mesh), Hv.x, color=:green)
scatter(x(mesh), y(mesh), Hv.y, color=:green)
scatter(x(mesh), y(mesh), divHv.values, color=:red)
scatter(x(mesh), y(mesh), divHv.vector.x, color=:red)
scatter(x(mesh), y(mesh), divHv.vector.y, color=:red)
scatter(xf(mesh), yf(mesh), divHv.face_vector.x, color=:red)
scatter(xf(mesh), yf(mesh), divHv.face_vector.y, color=:red)

scatter(x(mesh), y(mesh), p.values, color=:blue)
scatter!(xf(mesh), yf(mesh), pf.values, color=:red)

scatter(x(mesh), y(mesh), ∇p.x, color=:green)
scatter(x(mesh), y(mesh), ∇p.y, color=:green)

scatter(x(mesh), y(mesh), U.x, color=:green)
scatter(x(mesh), y(mesh), U.y, color=:green)
scatter(xf(mesh), yf(mesh), Uf.x, color=:red)
scatter(xf(mesh), yf(mesh), Uf.y, color=:red)

scatter(x(mesh), y(mesh), D, color=:red)
scatter(x(mesh), y(mesh), rD.values, color=:red)
scatter(xf(mesh), yf(mesh), rDf.values, color=:red)
