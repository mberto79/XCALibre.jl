using XCALibre

n_vertical      = 20 #400
n_horizontal1   = 200 #500
n_horizontal2   = 200 #800

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(2.0,0.0,0.0)
p4 = Point(0.0,0.2,0.0)
p5 = Point(1.0,0.2,0.0)
p6 = Point(2.0,0.2,0.0)

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
patch3 = Patch(:wall, [1,2])
patch4 = Patch(:top,    [3,4])
patches = [patch1, patch2, patch3, patch4]

blocks = [b1, b2]
builder = MeshBuilder2D(points, edges, patches, blocks)
mesh = generate!(builder)
mesh_new = XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64)

# Set up case for flat plate 

velocity = [0.2, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_new
    )

BCs = assign(
    region = mesh_new,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Extrapolated(:top)
        ],
        p = [
            Zerogradient(:outlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Extrapolated(:top)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes()
)


solvers = (
    U = SolverSetup(
        solver = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.8,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.2,
        rtol = 1e-2
    )
)

runtime = Runtime(iterations=2000, write_interval=1000, time_step=1)

# hardware = Hardware(backend=CUDABackend(), workgroup=32)
hardware = Hardware(backend=CPU(), workgroup=1024)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs