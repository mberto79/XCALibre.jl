using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

mesh_dev = mesh
# mesh_dev = adapt(CUDABackend(), mesh)  # Uncomment this if using GPU

# Inlet conditions
Umag = 0.5
velocity = [Umag, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh # mesh_dev  # use mesh_dev for GPU backend
    )

BCs = assign(
    region=mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Neumann(:outlet, 0.0),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)
        ],
        p = [
            Neumann(:inlet, 0.0),
            Dirichlet(:outlet, 0.0),
            Neumann(:wall, 0.0),
            Symmetry(:top)
        ]
    )
)

schemes = (
    U = set_schemes(divergence = Linear),
    # U = set_schemes(divergence = Upwind),
    p = set_schemes()
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver = Bicgstab(), # Bicgstab(), Gmres()
        smoother = JacobiSmoother(domain=mesh_dev, loops=5, omega=2/3),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.8,
        rtol = 1e-1,
    ),
    p = set_solver(
        model.momentum.p;
        solver = Cg(), # Bicgstab(), Gmres()
        smoother = JacobiSmoother(domain=mesh_dev, loops=5, omega=2/3),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.2,
        rtol = 1e-2,
    )
)

runtime = set_runtime(
    iterations=500, time_step=1, write_interval=500)

hardware = set_hardware(backend=CPU(), workgroup=1024)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing

residuals = run!(model, config)

inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)

@test outlet ≈ 0.5*inlet atol=0.1*Umag