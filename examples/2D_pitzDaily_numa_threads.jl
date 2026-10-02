# Step 0. Load libraries
using XCALibre
using ThreadPinning

# Multithreaded CPU run with NUMA placement by first touch. It pays on multi-socket nodes;
# results are the same as without it. Start Julia with several threads, e.g. `julia -t 8`.
pinthreads(:cores)  # threads must stay on their cores for placement to hold

# Step 1. Define mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "OF_pitzDaily/polyMesh"

mesh_file = joinpath(grids_dir, grid)
mesh = FOAM3D_mesh(mesh_file)

# Step 2. Select backend and setup hardware
backend = CPU(static=true); workgroup = AutoTune()  # chunk c of every loop runs on thread c
activate_multithread(backend; first_touch=true)     # equations and fields are placed from now on
mesh = first_touch(mesh)                            # place the mesh itself, before the model

hardware = Hardware(backend=backend, workgroup=workgroup)

mesh_dev = mesh

# Step 3. Flow conditions
velocity = [10.0, 0.0, 0.0]
nu = 1e-3

# Step 4. Define physics
model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

wall_patches = [:upperWall, :lowerWall]

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Empty(:frontAndBack),
            Wall.(wall_patches, Ref([0.0, 0.0, 0.0]))...
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Empty(:frontAndBack),
            Wall.(wall_patches, Ref(0.0))...
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind, gradient=Gauss),
    p = Schemes(gradient=Gauss)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 0.1
    ),
    p = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 0.01
    )
)

# Step 5. Specify runtime requirements
runtime = Runtime(iterations=500, time_step=1, write_interval=500)

# Step 6. Construct Configuration object
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# Step 7. Initialise fields (initial guess)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# Step 8. Run simulation
residuals = run!(model, config);
