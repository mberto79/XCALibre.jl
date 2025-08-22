
# Step 0. Load libraries
using XCALibre
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs
# using JLD2 # Uncomment to save objects
using ThreadPinning

pinthreads(:cores)

# Step 1. Define mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "OF_pitzDaily/polyMesh"

mesh_file = joinpath(grids_dir, grid)
mesh = FOAM3D_mesh(mesh_file)

# Step 2. Select backend and setup hardware
backend = CPU(); workgroup=AutoTune(); activate_multithread(backend)
# backend = CUDABackend(); workgroup = 32 # run on NVIDIA GPUs
# backend = ROCBackend(); workgroup = 32 # run on AMD GPUs

hardware = Hardware(backend=backend, workgroup=workgroup)

mesh_dev = adapt(hardware.backend, mesh)  # Move mesh to device (CPU or GPU) 

# Step 3. Flow conditions

U_inlet = 10.0
velocity = [U_inlet, 0.0, 0.0]
k_inlet = 2e-05
nu = 1e-5
Re = velocity[1]*0.1/nu
δt = 1e-05

# Step 4. Define physics

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    # turbulence = LES{Smagorinsky}(),
    turbulence = LES{KEquation}(),
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
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            Empty(:frontAndBack),
            Dirichlet.(wall_patches, Ref(0.0))...
        ],
        nut = [
            Extrapolated(:inlet),
            Extrapolated(:outlet, 0.0),
            Empty(:frontAndBack),
            Dirichlet.(wall_patches, Ref(0.0))...
        ]
    )
)

divScheme = Linear # Upwind Linear
schemes = (
    U = Schemes(time=Euler, divergence=divScheme, gradient=Gauss),
    p = Schemes(time=Euler, divergence=divScheme, gradient=Gauss),
    k = Schemes(time=Euler, divergence=divScheme, gradient=Gauss)
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        atol = 1e-5,
        rtol = 0.0;
        relax       = 1.0,
    ),
    p = SolverSetup(
        solver      = Cg(), # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        atol = 1e-5,
        rtol = 0.0;
        relax       = 0.9,
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        atol = 1e-5,
        rtol = 0.0;
        relax       = 1.0,
    ),
)

# Step 8. Specify runtime requirements
runtime = Runtime(
    # iterations=1, time_step=δt, write_interval=100)
    iterations=500, time_step=δt, write_interval=100)

# Step 9. Construct Configuration object
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# Step 10. Initialise fields (initial guess)
initial_velocity = [0.0, 0.0, 0.0]
initialise!(model.momentum.U, initial_velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.nut, 0.0)

# Step 11. Run simulation to compile
residuals = run!(model, config, inner_loops=2, ncorrectors=0, output=OpenFOAM());
