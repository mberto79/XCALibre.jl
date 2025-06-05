using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid = "bfs_unv_tet_10mm.unv"
grid = "bfs_unv_tet_15mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV3D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh3
# mesh_dev = adapt(CUDABackend(), mesh)  # Uncomment this if using GPU

# Inlet conditions
Umag = 0.5
velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh # mesh_dev  # use mesh_dev for GPU backend
    )
    
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Neumann(:outlet, 0.0),
    Symmetry(:top),
    Neumann(:sides, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0),
    Neumann(:sides, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Gauss),
    p = set_schemes(gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = Bicgstab(), #Cg(), Bicgstab(), Gmres(), 
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2,
    )
)

runtime = set_runtime(
    iterations=100, time_step=1, write_interval=100)

hardware = set_hardware(backend=CPU(), workgroup=1024)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing

residuals = run!(model, config)

top = boundary_average(:top, model.momentum.U, config)
outlet = boundary_average(:outlet, model.momentum.U, config)

@test Umag ≈ top[1] atol=0.1*Umag
@test 0.5*Umag ≈ outlet[1] atol=0.1*Umag