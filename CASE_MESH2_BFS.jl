using Plots

using XCALibre.Mesh2D
using XCALibre.Plotting
using XCALibre.Discretise
using XCALibre.Calculate
using XCALibre.Models
using XCALibre.Solvers

using Krylov

n_vertical      = 20 #400
n_horizontal1   = 25 #500
n_horizontal2   = 20 #800

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
# p2 = Point(1.0,0.0,0.0)
p3 = Point(2.0,0.0,0.0)
# p3 = Point(1.5,0.0,0.0)
p4 = Point(0.0,1.0,0.0)
p5 = Point(1.0,1.0,0.0)
# p5 = Point(1.0,1.0,0.0)
p6 = Point(2.0,1.0,0.0)
# p6 = Point(1.5,0.7,0.0)
points = [p1, p2, p3, p4, p5, p6]

scatter(points)

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

BCs = (
    Dirichlet(:inlet, 100.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
    # Dirichlet(:bottom, 50.0),
    # Dirichlet(:top, 50.0)
)


blocks = [b1, b2]
builder = MeshBuilder2D(points, edges, patches, blocks)
mesh = generate!(builder)

plot_mesh!(mesh)

blocks = [b2, b1]
builder = MeshBuilder2D(points, edges, patches, blocks)
mesh = generate!(builder)
plot_mesh!(mesh)

phi = ScalarField(mesh)
J = 1.0
model = Diffusion(
    Laplacian{Linear}(J, phi),
    0.0
    )

equation = Equation(mesh)
generate_boundary_conditions!(mesh, model, BCs)


setup = SolverSetup(
    iterations  = 100,
    solver      = GmresSolver,
    tolerance   = 1e-6,
    relax       = 1.0,
    itmax       = 100
    )

clear!(phi)
@time solve_system!(equation, model, BCs, setup)

scatter(x(mesh), y(mesh), phi.values, color=:blue)

# HORIZONALLY CONNECTED BLOCKS 

n_vertical1     = 20 #400
n_vertical2     = 25 #500
n_horizontal    = 20 #800

p1 = Point(0.0,0.0,0.0)
p2 = Point(2.0,0.0,0.0)
p3 = Point(0.0,0.5,0.0)
p4 = Point(2.0,0.5,0.0)
p5 = Point(0.0,1.0,0.0)
p6 = Point(2.0,1.0,0.0)
points = [p1, p2, p3, p4, p5, p6]

scatter(points)

# Edges in x-direction
e1 = line!(points,1,2,n_horizontal1)
e2 = line!(points,3,4,n_horizontal1)
e3 = line!(points,5,6,n_horizontal1)

# Edges in y-direction
e4 = line!(points,1,3,n_vertical1)
e5 = line!(points,2,4,n_vertical1)
e6 = line!(points,3,5,n_vertical2)
e7 = line!(points,4,6,n_vertical2)
edges = [e1, e2, e3, e4, e5, e6, e7]

b1 = quad(edges, [1,2,4,5])
b2 = quad(edges, [2,3,6,7])
blocks = [b1, b2]

patch1 = Patch(:inlet,  [4,6])
patch2 = Patch(:outlet, [5,7])
patch3 = Patch(:bottom, [1])
patch4 = Patch(:top,    [3])
patches = [patch1, patch2, patch3, patch4]

BCs = (
    Dirichlet(:inlet, 100.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
    # Dirichlet(:bottom, 50.0),
    # Dirichlet(:top, 50.0)
)


blocks = [b1, b2]
blocks = [b2, b1]
builder = MeshBuilder2D(points, edges, patches, blocks)
mesh = generate!(builder)

plot_mesh!(mesh)

phi = ScalarField(mesh)
J = 1.0
model = Diffusion(
    Laplacian{Linear}(J, phi),
    0.0
    )

equation = Equation(mesh)
generate_boundary_conditions!(mesh, model, BCs)


setup = SolverSetup(
    iterations  = 100,
    solver      = GmresSolver,
    tolerance   = 1e-6,
    relax       = 1.0,
    itmax       = 100
    )

clear!(phi)
@time solve_system!(equation, model, BCs, setup)

scatter(x(mesh), y(mesh), phi.values, color=:blue)
