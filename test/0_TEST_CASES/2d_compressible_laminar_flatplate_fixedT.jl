using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_laminar.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

Umag = 0.2
velocity = [Umag, 0.0, 0.0]
nu = 1e-4
Re = velocity[1]*1/nu
cp = 1005.0
gamma = 1.4
Pr = 0.7

model = Physics(
    time = Steady(),
    fluid =  Fluid{WeaklyCompressible}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=288.15),
    domain = mesh_dev # mesh_dev  # use mesh_dev for GPU backend
    )

BCs = assign(
    region=mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 100000.0),
            Wall(:wall),
            Symmetry(:top)
        ],
        h = [
            FixedTemperature(:inlet, T=300.0, model=model.energy),
            Extrapolated(:outlet),
            FixedTemperature(:wall, T=310.0, model=model.energy),
            Symmetry(:top)
        ],
    )
)

schemes = (
    U = set_schemes(divergence=Linear),
    p = set_schemes(divergence=Linear),
    h = set_schemes(divergence=Linear)
)

solvers = (
    U = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-1
    ),
    p = set_solver(
        region=mesh_dev,
        solver      = Cg(), #Gmres(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2
    ),
    h = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-1
    )
)

runtime = set_runtime(iterations=100, write_interval=100, time_step=1)

hardware = set_hardware(backend=backend, workgroup=workgroup)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 100000.0) === nothing
@test initialise!(model.energy.T, 300.0) === nothing

residuals = run!(model, config)

inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)
top = boundary_average(:top, model.momentum.U, BCs.U, config)

@test Umag ≈ inlet[1]
@test Umag ≈ outlet[1] atol = 0.07
@test Umag ≈ top[1] atol = 0.023