using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Inlet conditions

velocity = [0.5, 0.0, 0.0]
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
    fluid = Fluid{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=288.15),
    domain = mesh_dev
    )

BCs =assign(
    region =  mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Neumann(:outlet, 0.0),
            Wall(:cylinder, noSlip),
            Neumann(:bottom, 0.0),
            Neumann(:top, 0.0)
        ],
        p = [
            Neumann(:inlet, 0.0),
            Dirichlet(:outlet, pressure),
            Neumann(:cylinder, 0.0),
            Neumann(:bottom, 0.0),
            Neumann(:top, 0.0)
        ],
        h = [
            FixedTemperature(:inlet, T=temp, Enthalpy(cp=cp, Tref=288.15)),
            Neumann(:outlet, 0.0),
            FixedTemperature(:cylinder, T=330.0, Enthalpy(cp=cp, Tref=288.15)),
            Neumann(:bottom, 0.0),
            Neumann(:top, 0.0)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1,
        rtol = 1e-4
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        limit = (1000, 1000000),
        rtol = 1e-4
    ),
    h = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1,
        rtol = 1e-4
    )
)

schemes = (
    rho = Schemes(time=Euler),
    U = Schemes(divergence=Upwind, gradient=Midpoint, time=Euler),
    p = Schemes(gradient=Midpoint, time=Euler),
    h = Schemes(divergence=Upwind, gradient=Midpoint, time=Euler)
)

runtime = Runtime(iterations=1000, write_interval=100, time_step=0.01)

hardware = Hardware(backend=backend, workgroup=workgroup)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)
initialise!(model.fluid.rho, pressure/(R*temp))

residuals = run!(model, config, ncorrectors=1)