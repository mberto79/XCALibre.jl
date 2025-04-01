using XCALibre
# using CUDA

# backwardFacingStep_2mm, 5mm or 10mm
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); activate_multithread(backend); workgroup=1024
# backend = CUDABackend(); workgroup=32
mesh_dev = adapt(backend, mesh)

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
    # Wall(:top, [0.0, 0.0, 0.0])
    # Neumann(:top, 0.0),
    Symmetry(:top)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Wall(:wall, 0.0),
    # Neumann(:top, 0.0)
    Symmetry(:top)
)

schemes = (
    # U = set_schemes(divergence = Linear, limiter=MFaceBased(model.domain)),
    # U = set_schemes(divergence = Linear),
    U = set_schemes(divergence = Linear),
    p = set_schemes()
    # p = set_schemes(limiter=FaceBased(model.domain))
    # p = set_schemes(limiter=MFaceBased(model.domain))
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # ILU0GPU, Jacobi, DILU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=8, omega=1),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver, CgSolver
        preconditioner = Jacobi(), # IC0GPU, Jacobi, DILU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=8, omega=1),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2
    )
)

runtime = set_runtime(
    iterations=2000, time_step=1.0, write_interval=1000)
    # iterations=1, time_step=1, write_interval=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=backend, workgroup=workgroup)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)


@time residuals = run!(model, config)

using Plots
iterations = runtime.iterations
plot(yscale=:log10, ylims=(1e-8,1e-1))
plot!(1:iterations, residuals.Ux, label="Ux")
plot!(1:iterations, residuals.Uy, label="Uy")
plot!(1:iterations, residuals.Uz, label="Uz")
plot!(1:iterations, residuals.p, label="p")
