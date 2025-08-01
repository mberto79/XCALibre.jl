using XCALibre
# using CUDA
# using ThreadPinning

# pinthreads(:cores)

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid = "bfs_unv_tet_4mm.unv"
# grid = "bfs_unv_tet_5mm.unv"
grid = "bfs_unv_tet_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_5mm.unv"
# mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_4mm.unv"
# mesh_file = "bfs_unv_tet_5mm.unv"

# mesh_file = "/Users/hmedi/Desktop/BFS_GRIDS/bfs_unv_tet_4mm.unv"
# mesh_file = "/home/humberto/Desktop/BFS_GRIDS/bfs_unv_tet_5mm.unv"
@time mesh = UNV3D_mesh(mesh_file, scale=0.001) # 36 sec
# @time mesh = UNV3D_mesh(mesh_file, scale=0.001, float_type=Float32)

# backend = CUDABackend(); workgroup = 32
backend = CPU(static=static); workgroup = AutoTune(); activate_multithread(backend)

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
            Extrapolated(:sides),
            Extrapolated(:top)
        ],
        p = [
            Extrapolated(:inlet),
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
        rtol = 0.1,
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

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs
)

GC.gc(false)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

# Now get timing information

runtime = Runtime(iterations=500, write_interval=500, time_step=1)
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)
configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs
)

GC.gc(false)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

@time residuals = run!(model, config, output=OpenFOAM(), ncorrectors=0)

# using Plots
# iterations = runtime.iterations
# plot(yscale=:log10, ylims=(1e-7,1e-1))
# plot!(1:iterations, residuals.Ux, label="Ux")
# plot!(1:iterations, residuals.Uy, label="Uy")
# plot!(1:iterations, residuals.Uz, label="Uz")
# plot!(1:iterations, residuals.p, label="p")