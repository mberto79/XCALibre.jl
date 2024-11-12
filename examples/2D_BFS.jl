using XCALibre
# using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_dev = adapt(CUDABackend(), mesh)
mesh_dev = mesh

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
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    # Dirichlet(:wall, [0.0, 0.0, 0.0]),
    # Dirichlet(:top, [0.0, 0.0, 0.0]),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Wall(:top, [0.0, 0.0, 0.0])
    # Symmetry(:top)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
    # Symmetry(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence = Linear),
    # U = set_schemes(divergence = Upwind),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # ILU0GPU
        smoother=JacobiSmoother(domain=mesh_dev, loops=5, omega=2/3),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 0.1
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # IC0GPU
        smoother=JacobiSmoother(domain=mesh_dev, loops=5, omega=2/3),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 0.01
    )
)

runtime = set_runtime(
    iterations=1000, time_step=1, write_interval=1000)
    # iterations=1, time_step=1, write_interval=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

@time residuals = run!(model, config)

using Plots
iterations = runtime.iterations
plot(yscale=:log10, ylims=(1e-7,1e-1))
plot!(1:iterations, residuals.Ux, label="Ux")
plot!(1:iterations, residuals.Uy, label="Uy")
plot!(1:iterations, residuals.Uz, label="Uz")
plot!(1:iterations, residuals.p, label="p")