# using Plots
using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_25mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

# Inlet conditions
Umag = 0.5
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-4
Re = (0.2*velocity[1])/nu
gamma = 1.4
cp = 1005.0
R = 287.0
temp = 300.0
pressure = 100000
Pr = 0.7

model = Physics(
    time = Transient(),
    fluid = Fluid{WeaklyCompressible}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=288.15),
    domain = mesh_dev # mesh_dev  # use mesh_dev for GPU backend
    )

BCs = assign(
    region=mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Neumann(:outlet, 0.0),
            Wall(:cylinder, noSlip),
            Symmetry(:bottom),
            Symmetry(:top)
        ],
        p = [
            Neumann(:inlet, 0.0),
            Dirichlet(:outlet, pressure),
            Wall(:cylinder),
            Symmetry(:bottom),
            Symmetry(:top)
        ],
        h = [
            FixedTemperature(:inlet, T=300.0, model=model.energy),
            Neumann(:outlet, 0.0),
            FixedTemperature(:cylinder, T=330.0, model=model.energy),
            Symmetry(:bottom),
            Symmetry(:top)
        ]
    )
)

solvers = (
    U = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1,
        rtol = 1e-4
            
        ),
    p = set_solver(
        region=mesh_dev,
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        limit = (1000, 1000000),
        rtol = 1e-4
    ),
    h = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1,
        rtol = 1e-4
    )
)

schemes = (
    rho = set_schemes(time=Euler),
    U = set_schemes(divergence=Upwind, gradient=Midpoint, time=Euler),
    p = set_schemes(gradient=Midpoint, time=Euler),
    h = set_schemes(divergence=Upwind, gradient=Midpoint, time=Euler)
)

runtime = set_runtime(iterations=100, write_interval=100, time_step=0.01)

hardware = set_hardware(backend=backend, workgroup=workgroup)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, pressure) === nothing
@test initialise!(model.energy.T, temp) === nothing
@test initialise!(model.fluid.rho, pressure/(R*temp)) === nothing

residuals = run!(model, config)

inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)
top = boundary_average(:top, model.momentum.U, BCs.U, config)
bottom = boundary_average(:bottom, model.momentum.U, BCs.U, config)

@test Umag ≈ inlet[1]
@test Umag ≈ outlet[1] atol = 0.1*Umag
@test Umag ≈ top[1] atol = 0.1*Umag
@test top[1] ≈ bottom[1] atol = 0.1*Umag