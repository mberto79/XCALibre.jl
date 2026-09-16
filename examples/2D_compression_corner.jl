using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "compression_corner_2d_32_64_SF4.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024
# backend = CUDABackend(); workgroup = 32
# backend = ROCBackend(); workgroup = 32
mesh_dev = adapt(backend, mesh)

# Inlet/reference conditions (Mach 2 supersonic inflow)
nu = 0.0
gamma = 1.4
cp = 1005.0
R = cp*(1.0 - (1.0/gamma))
cv = cp - R
temp = 1000.0
Tref = 0.0
pressure = 100000.0
Pr = 0.7

M = 2
a = sqrt(gamma*R*temp)
rho = pressure/(R*temp)
Umag = M*a
velocity = [Umag, 0.0, 0.0]

model = Physics(
    time = Steady(),
    fluid = Fluid{Compressible}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=Tref),
    # energy = Energy{InternalEnergy}(Tref=Tref),
    domain = mesh_dev
    )

boundaries = assign(
    region = mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Zerogradient(:top),
            Slip(:wall)
        ],
        p = [
            Dirichlet(:inlet, pressure),
            Zerogradient(:outlet),
            Zerogradient(:top),
            Zerogradient(:wall)
        ],
        he = [
            # Use IEnergy(cv=cv, ...) below when energy = Energy{InternalEnergy}
            FixedTemperature(:inlet, T=temp, Enthalpy(cp=cp, Tref=Tref)),
            # FixedTemperature(:inlet, T=temp, IEnergy(cv=cv, Tref=Tref)),
            Zerogradient(:outlet),
            Zerogradient(:top),
            Zerogradient(:wall)
        ]
    )
)

atol = 1e-2
rtol = 0.0
solvers = (
    U = SolverSetup(
        solver = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-10,
        relax = 0.7,
        rtol = rtol,
        atol = atol
    ),
    p = SolverSetup(
        solver = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-10,
        relax = 0.3,
        limit = (0.5*pressure, 5*pressure),
        rtol = rtol,
        atol = atol
    ),
    he = SolverSetup(
        solver = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-10,
        relax = 0.3,
        limit = (800.0, 3000.0),
        rtol = rtol,
        atol = atol
    )
)

divergence = Upwind # BoundedUpwind
schemes = (
    U = Schemes(time=SteadyState, divergence=divergence),
    p = Schemes(time=SteadyState, divergence=Upwind),
    he = Schemes(time=SteadyState, divergence=divergence)
)

runtime = Runtime(iterations=10000, write_interval=100, time_step=1)
hardware = Hardware(backend=backend, workgroup=workgroup)

config = Configuration(;
    solvers, schemes, runtime, hardware, boundaries)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)

residuals = run!(model, config, output=VTK())
