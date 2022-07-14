using Plots
using Krylov

# using FVM_1D

using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.Discretise
using FVM_1D.Calculate
using FVM_1D.Models
using FVM_1D.Solvers
using FVM_1D.VTK
using FVM_1D.UNV


function generate_mesh(n_horizontal, n_vertical)

    p1 = Point(0.0,0.0,0.0)
    p2 = Point(0.5,0.0,0.0)
    # p2 = Point(0.5,-0.2,0.0)
    p3 = Point(0.0,0.1,0.0)
    p4 = Point(0.5,0.1,0.0)
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

velocity = [1.0, 0.0, 0.0]
nu = 1e-3
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

setup_U = SolverSetup(
    solver      = BicgstabSolver,
    relax       = 0.7,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = BicgstabSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    relax       = 0.3,
    itmax       = 100,
    rtol        = 1e-2
)

#SymmlqSolver, MinresSolver - did not work!
n_vertical      = 40 #40 
n_horizontal    = 200 #200 
mesh = generate_mesh(n_horizontal, n_vertical)

GC.gc()

U = VectorField(mesh)
p = ScalarField(mesh)

iterations = 500
Rx, Ry, Rp = isimple!(
    mesh, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations)

write_vtk("results", mesh, ("U", U), ("p", p))

# plotly(size=(400,400), markersize=1, markerstrokewidth=1)
niterations = length(Rx)
plot(collect(1:niterations), Rx[1:niterations], yscale=:log10)
plot!(collect(1:niterations), Ry[1:niterations], yscale=:log10)
plot!(collect(1:niterations), Rp[1:niterations], yscale=:log10)

scatter(x(mesh), y(mesh), ux.values, color=:red)
scatter(x(mesh), y(mesh), uy.values, color=:red)

scatter(x(mesh), y(mesh), Hv.x, color=:green)
scatter(x(mesh), y(mesh), U.x + ∇p.x.*rD.values, color=:blue)

scatter(x(mesh), y(mesh), Hv.y, color=:green)
scatter(x(mesh), y(mesh), divHv.values, color=:red)
scatter(x(mesh), y(mesh), divHv.vector.x, color=:red)
scatter(x(mesh), y(mesh), divHv.vector.y, color=:red)
scatter(xf(mesh), yf(mesh), divHv.face_vector.x, color=:blue)
scatter(xf(mesh), yf(mesh), divHv.face_vector.y, color=:blue)

scatter(x(mesh), y(mesh), p.values, color=:blue)
scatter!(xf(mesh), yf(mesh), pf.values, color=:red)

scatter(x(mesh), y(mesh), ∇p.x, color=:green)
scatter(x(mesh), y(mesh), ∇p.y, color=:green)

scatter(x(mesh), y(mesh), U.x, color=:green)
scatter(x(mesh), y(mesh), U.y, color=:green)
scatter(xf(mesh), yf(mesh), Uf.x, color=:red)
scatter(xf(mesh), yf(mesh), Uf.y, color=:red)

scatter(x(mesh), y(mesh), rD.values, color=:red)
scatter(xf(mesh), yf(mesh), rDf.values, color=:red)

scatter(x(mesh), y(mesh), mdot.values, color=:red)
scatter(xf(mesh), yf(mesh), mdotf.values, color=:red)

scatter(mesh.nodes)
