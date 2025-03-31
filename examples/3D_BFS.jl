using Plots 
# using ThreadPinning
using XCALibre
using CUDA

# pinthreads(:cores)

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid = "bfs_unv_tet_4mm.unv"
# grid = "bfs_unv_tet_5mm.unv"
grid = "bfs_unv_tet_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

# mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_5mm.unv"
# mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_4mm.unv"
# mesh_file = "bfs_unv_tet_5mm.unv"

# mesh_file = "/Users/hmedi/Desktop/BFS_GRIDS/bfs_unv_tet_4mm.unv"
mesh_file = "/home/humberto/Desktop/BFS_GRIDS/bfs_unv_tet_5mm.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

# workgroup = cld(length(mesh.cells), Threads.nthreads())
# backend = CPU(); activate_multithread(backend)
# backend = CPU()
# activate_multithread1()
workgroup = 32
backend = CUDABackend()

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

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, noSlip),
    Neumann(:sides, 0.0),
    Neumann(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Wall(:wall, 0.0),
    Neumann(:sides, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Jacobi # ILU0GPU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=10, omega=2/3),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 0.1
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #NormDiagonal(), IC0GPU, Jacobi
        # smoother=JacobiSmoother(domain=mesh_dev, loops=10, omega=2/3),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 0.1,
        itmax = 1000
    )
)

schemes = (
    U = set_schemes(time=SteadyState, divergence=Upwind, gradient=Midpoint),
    p = set_schemes(time=SteadyState, gradient=Midpoint)
)

hardware = set_hardware(backend=backend, workgroup=workgroup)

# Run first to pre-compile

runtime = set_runtime(iterations=1, write_interval=1, time_step=1)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

# Now get timing information

runtime = set_runtime(iterations=500, write_interval=100, time_step=1)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

@time residuals = run!(model, config)

# iterations = runtime.iterations
# plot(yscale=:log10, ylims=(1e-7,1e-1))
# plot!(1:iterations, residuals.Ux, label="Ux")
# plot!(1:iterations, residuals.Uy, label="Uy")
# plot!(1:iterations, residuals.Uz, label="Uz")
# plot!(1:iterations, residuals.p, label="p")