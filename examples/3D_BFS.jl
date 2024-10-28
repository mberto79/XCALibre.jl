using Plots
using XCALibre
using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "bfs_unv_tet_4mm.unv"
grid = "bfs_unv_tet_5mm.unv"
grid = "bfs_unv_tet_10mm.unv"

mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_5mm.unv"
mesh_file = "/home/humberto/foamCases/jCFD_benchmarks/3D_BFS/bfs_unv_tet_4mm.unv"

# mesh_file = joinpath(grids_dir, grid)

@time mesh = UNV3D_mesh(mesh_file, scale=0.001)

# @time mesh = FOAM3D_mesh(mesh_file, scale=0.001, integer_type=Int64, float_type=Float64)

mesh_dev = adapt(CUDABackend(), mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )
    

@assign! model momentum U (
    # Dirichlet(:inlet, velocity),
    # Neumann(:outlet, 0.0),0.0]),
    # Dirichlet(:sides, [0.0, 0.0, 0.0])
    Dirichlet(:inlet, velocity),
    Wall(:wall, [0.0, 0.0, 0.0]),
    # Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
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
    U = set_schemes(divergence=Upwind, gradient=Orthogonal),
    p = set_schemes(gradient=Orthogonal)
    # p = set_schemes()
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, #CgSolver, # BicgstabSolver, GmresSolver, #CgSolver
        preconditioner = Jacobi(), # Jacobi ILU0GPU
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 0.1,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Jacobi IC0GPU
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 0.1,
    )
)

runtime = set_runtime(
    iterations=1500, time_step=1, write_interval=500)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config, limit_gradient=true)
# residuals = run!(model, config, limit_gradient=false)

xrange = 1:runtime.iterations
plot(; xlims=(0,runtime.iterations), ylims=(1e-7,0.2))
plot!(xrange, residuals.Ux, yscale=:log10, label="Ux")
plot!(xrange, residuals.Uy, yscale=:log10, label="Uy")
plot!(xrange, residuals.Uz, yscale=:log10, label="Uz")
plot!(xrange, residuals.p, yscale=:log10, label="p")
