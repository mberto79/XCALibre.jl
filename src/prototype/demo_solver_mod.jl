
# Step 0. Load libraries
using XCALibre
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs


# Step 1. Define mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001) # convert mesh

# Step 2. Select backend and setup hardware
backend = CPU()
# backend = CUDABackend() # ru non NVIDIA GPUs
# backend = ROCBackend() # run on AMD GPUs

hardware = set_hardware(backend=backend, workgroup=4)
# hardware = set_hardware(backend=backend, workgroup=32) # use for GPU backends

mesh_dev = mesh # use this line to run on CPU
# mesh_dev = adapt(backend, mesh)  # Uncomment to run on GPU 

# Step 3. Flow conditions
velocity = [1.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

# Step 4. Define physics
model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

# Step 5. Define boundary conditions
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Wall(:top, [0.0, 0.0, 0.0]),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

# Step 6. Choose discretisation schemes
schemes = (
    U = set_schemes(divergence = Linear),
    p = set_schemes() # no input provided (will use defaults)
)

# Step 7. Set up linear solvers and preconditioners
solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = Bicgstab(), # Options: Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 0.01,#1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = Cg(), # Options: Cg(), Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 0.01,#1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    )
)

# Step 8. Specify runtime requirements
runtime = set_runtime(iterations=2000, time_step=1, write_interval=2000)
#runtime = set_runtime(iterations=1, time_step=1, write_interval=-1) # hide

# Step 9. Construct Configuration object
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# Step 10. Initialise fields (initial guess)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# Step 11. Run simulation
residuals = run!(model, config);

U = VectorField(mesh)
Uf = FaceVectorField(mesh)

interpolate!(Uf, Hv, config)

# Step 12. Post-process
# pwd() # find active directory where the file "iteration_002000.vtk" was saved