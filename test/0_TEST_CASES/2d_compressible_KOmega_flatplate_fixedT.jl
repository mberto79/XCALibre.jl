using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_laminar.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2
# mesh_dev = adapt(CUDABackend(), mesh)  # Uncomment this if using GPU

# Inlet conditions
Umag = 10
velocity = [Umag, 0.0, 0.0]
nu = 1e-4
Re = velocity[1]*1/nu
cp = 1005.0
gamma = 1.4
Pr = 0.7

k_inlet = 0.375
ω_inlet = 1000

model = Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{KOmega}(),
    energy = Energy{SensibleEnthalpy}(Tref=288.15),
    domain = mesh # mesh_dev  # use mesh_dev for GPU backend
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Symmetry(:top, 0.0)
)

 @assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 100000.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy h (
    FixedTemperature(:inlet, T=300.0, model=model.energy),
    Neumann(:outlet, 0.0),
    FixedTemperature(:wall, T=310.0, model=model.energy),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    # KWallFunction(:wall),
    # KWallFunction(:top)
    Dirichlet(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    Neumann(:top, 0.0)
    # Dirichlet(:wall, ω_wall), 
    # Dirichlet(:top, ω_wall)
)

@assign! model turbulence nut (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0), 
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Linear),
    p = set_schemes(divergence=Linear),
    h = set_schemes(divergence=Linear),
    k = set_schemes(divergence=Upwind),
    omega = set_schemes(divergence=Upwind)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
    ),
    h = set_solver(
        model.energy.h;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-4
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #ILU0(),
        convergence = 1e-7,
        relax       = 0.8,
        atol = 1e-6
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #ILU0(),
        convergence = 1e-7,
        relax       = 0.8,
        atol = 1e-6
    )
)

runtime = set_runtime(iterations=200, write_interval=200, time_step=1)

hardware = set_hardware(backend=CPU(), workgroup=cld(length(mesh.faces), 4))
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 100000.0) === nothing
@test initialise!(model.energy.T, 300.0) === nothing
@test initialise!(model.turbulence.k, k_inlet) === nothing
@test initialise!(model.turbulence.omega, ω_inlet) === nothing
@test initialise!(model.turbulence.nut, k_inlet/ω_inlet) === nothing

residuals = run!(model, config)

inlet = boundary_average(:inlet, model.momentum.U, config)
outlet = boundary_average(:outlet, model.momentum.U, config)
top = boundary_average(:top, model.momentum.U, config)

@test Umag ≈ inlet[1]
@test Umag ≈ outlet[1] atol = 0.75
@test Umag ≈ top[1] atol = 0.15