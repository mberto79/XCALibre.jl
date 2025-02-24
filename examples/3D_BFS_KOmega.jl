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

mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_5mm.unv"
# mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_5mm_with_BL.unv"
# mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_4mm.unv"
# mesh_file = "bfs_unv_tet_5mm.unv"

# mesh_file = "/Users/hmedi/Desktop/BFS_GRIDS/bfs_unv_tet_4mm.unv"
# mesh_file = "/home/humberto/Desktop/BFS_GRIDS/bfs_unv_tet_5mm.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

# workgroup = cld(length(mesh.cells), Threads.nthreads())
# backend = CPU(); activate_multithread(backend)
# backend = CPU()
# activate_multithread1()
workgroup = 32
backend = CUDABackend()

mesh_dev = adapt(backend, mesh)

# Inlet conditions
nu = 1e-3
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
kInlet = velocity[1]*0.01
omegaInlet = kInlet/(100*nu)
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, noSlip),
    Neumann(:sides, 0.0),
    Symmetry(:top)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:sides, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, kInlet),
    Neumann(:outlet, 0.0),
    KWallFunction(:wall),
    Neumann(:sides, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, omegaInlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    Neumann(:sides, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence nut (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    NutWallFunction(:wall),
    Neumann(:sides, 0.0),
    Neumann(:top, 0.0)
)

sloops = 10; omega = 2/3
rtol = 1e-1
prtol = 1e-2
solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Jacobi(), ILU0GPU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=sloops, omega=omega),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = rtol
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #NormDiagonal(), Jacobi, IC0GPU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=sloops, omega=omega),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = prtol
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Jacobi(), ILU0GPU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=sloops, omega=omega),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = rtol
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Jacobi(), ILU0GPU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=sloops, omega=omega),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = rtol
    )
)

default_schemes = (time=SteadyState, divergence=Upwind, gradient=Midpoint)
schemes = (
    U = set_schemes(; default_schemes...),
    p = set_schemes(; default_schemes...),
    k = set_schemes(; default_schemes...),
    omega = set_schemes(; default_schemes...),
)

hardware = set_hardware(backend=backend, workgroup=workgroup)

# Run first to pre-compile

runtime = set_runtime(iterations=1, write_interval=1, time_step=1)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, noSlip)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, kInlet)
initialise!(model.turbulence.omega, omegaInlet)
initialise!(model.turbulence.nut, kInlet/omegaInlet)


residuals = run!(model, config, ncorrectors=0)

# Now get timing information

runtime = set_runtime(iterations=500, write_interval=100, time_step=1)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, noSlip)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, kInlet)
initialise!(model.turbulence.omega, omegaInlet)
initialise!(model.turbulence.nut, kInlet/omegaInlet)

@time residuals = run!(model, config, ncorrectors=0)

# iterations = runtime.iterations
# plot(yscale=:log10, ylims=(1e-7,1e-1))
# plot!(1:iterations, residuals.Ux, label="Ux")
# plot!(1:iterations, residuals.Uy, label="Uy")
# plot!(1:iterations, residuals.Uz, label="Uz")
# plot!(1:iterations, residuals.p, label="p")