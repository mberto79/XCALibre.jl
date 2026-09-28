using XCALibre
using CUDA
# using ThreadPinning

# pinthreads(:cores)

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid = "bfs_unv_tet_4mm.unv"
# grid = "bfs_unv_tet_5mm.unv"
grid = "bfs_unv_tet_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

grids_dir = "/home/humberto/Desktop/BFS_GRIDS"
mesh = UNV3D_mesh(joinpath(grids_dir, "bfs_unv_tet_4mm.unv"), scale=0.001)

# backend = CUDABackend(); workgroup = 32
# backend = CPU(); workgroup = 1024; activate_multithread(backend)
backend = CPU(); workgroup = AutoTune()
activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Inlet conditions
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:wall, noSlip),
            Zerogradient(:sides), # faster!
            Zerogradient(:top)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Extrapolated(:sides),
            Extrapolated(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        # float_type = Float32,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Jacobi # ILU0GPU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=10, omega=2/3),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 0.1
    ),
    p = SolverSetup(
        # float_type = Float32,
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), #NormDiagonal(), IC0GPU, Jacobi
        # smoother=JacobiSmoother(domain=mesh_dev, loops=10, omega=2/3),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 0.01,
        itmax = 1000
    )
)

gradScheme = Gauss # Gauss # Midpoint
divScheme = Upwind # Upwind
schemes = (
    U = Schemes(time=SteadyState, divergence=divScheme, gradient=gradScheme),
    p = Schemes(time=SteadyState, gradient=gradScheme)
)

# Run first to pre-compile

runtime = Runtime(iterations=1, write_interval=1, time_step=1)
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(false)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config, output=OpenFOAM())

# Now get timing information

runtime = Runtime(iterations=100, write_interval=100, time_step=1)
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(false)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# @time residuals = run!(model, config, output=OpenFOAM(), ncorrectors=0)
@time residuals = run!(model, config, output=OpenFOAM(), ncorrectors=0)
