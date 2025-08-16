# Step 0. Load libraries
using XCALibre
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs


# Step 1. Define mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "OF_pitzDaily/polyMesh"

mesh_file = joinpath(grids_dir, grid)
mesh = FOAM3D_mesh(mesh_file)
@test typeof(mesh) <: Mesh3
@test length(mesh.cells) == 12225
@test length(mesh.faces) == 49180
@test length(mesh.nodes) == 25012

# Step 2. Select backend and setup hardware
backend = CPU(); workgroup=AutoTune()
# backend = CUDABackend(); workgroup = 32 # run on NVIDIA GPUs
# backend = ROCBackend(); workgroup = 32 # run on AMD GPUs

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(hardware.backend, mesh) 


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
    turbulence = LES{Smagorinsky}(), # Smagorinsky,
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

divScheme = Upwind # Upwind Linear
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
        relax = 1.0,
    ),
    p = SolverSetup(
        solver = Cg(), # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        atol = 1e-5,
        rtol = 0.0;
        relax = 1.0,
    ),
    k = SolverSetup(
        solver = Bicgstab(), # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        atol = 1e-5,
        rtol = 0.0;
        relax = 1.0,
    ),
)

# Step 8. Specify runtime requirements
runtime = Runtime(
    iterations=100, time_step=δt, write_interval=100)

# Step 9. Construct Configuration object
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# Step 10. Initialise fields (initial guess)
initial_velocity = [0.0, 0.0, 0.0]
initialise!(model.momentum.U, initial_velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.nut, 0.0)

# Step 11. Run simulation
residuals = run!(model, config, inner_loops=2, ncorrectors=0, output=OpenFOAM());

inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)

@test inlet[1] ≈ 10
@test outlet[1] ≈ 7.60 atol=0.1
