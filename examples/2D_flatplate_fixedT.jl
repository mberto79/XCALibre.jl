using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_laminar.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_dev = adapt(CUDABackend(), mesh)

velocity = [0.2, 0.0, 0.0]
nu = 1e-4
Re = velocity[1]*1/nu
cp = 1005.0
gamma = 1.4
Pr = 0.7

model = Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref = 288.15),
    domain = mesh
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

schemes = (
    U = set_schemes(divergence=Linear),
    p = set_schemes(divergence=Linear),
    h = set_schemes(divergence=Linear)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = Gmres(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
    ),
    h = set_solver(
        model.energy.h;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-4
    )
)

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=cld(length(mesh.cells),4))

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 100000.0)
initialise!(model.energy.T, 300.0)

residuals = run!(model, config)