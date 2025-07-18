using XCALibre
# using CUDA

# backwardFacingStep_2mm, 5mm or 10mm
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [0.0, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Extrapolated(:inlet),
            Wall(:wall, [0.0, 0.0, 0.0]),

            Extrapolated(:outlet),

            Wall(:top, [0.0, 0.0, 0.0])
            # Symmetry(:top)
        ],
        p = [

            Neumann(:inlet, -2.75),
            Wall(:wall),

            # Dirichlet(:outlet, 0.0),
            Extrapolated(:outlet),

            Wall(:top)
            # Symmetry(:top)
        ]
    )
)

schemes = (
    # U = Schemes(divergence = Linear, limiter=MFaceBased(model.domain)),
    # U = Schemes(divergence = Linear),
    U = Schemes(divergence = Upwind),
    p = Schemes()
    # p = Schemes(limiter=FaceBased(model.domain))
    # p = Schemes(limiter=MFaceBased(model.domain))
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # ILU0GPU, Jacobi, DILU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=8, omega=1),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres(), Cg()
        preconditioner = Jacobi(), # IC0GPU, Jacobi, DILU
        # smoother=JacobiSmoother(domain=mesh_dev, loops=8, omega=1),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2
    )
)

runtime = Runtime(
    iterations=100, time_step=1.0, write_interval=1)
    # iterations=1, time_step=1, write_interval=1)

# hardware = Hardware(backend=CUDABackend(), workgroup=32)
hardware = Hardware(backend=backend, workgroup=workgroup)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)
# config = adapt(CUDABackend(), config)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# residuals = run!(model, config, pref=0)
residuals = run!(model, config)
# @time residuals = run!(model, config)

# GC.gc()

# initialise!(model.momentum.U, velocity)
# initialise!(model.momentum.p, 0.0)

# @profview residuals = run!(model, config)

# using Plots
# iterations = runtime.iterations
# plot(yscale=:log10, ylims=(1e-8,1e-1))
# plot!(1:iterations, residuals.Ux, label="Ux")
# plot!(1:iterations, residuals.Uy, label="Uy")
# plot!(1:iterations, residuals.Uz, label="Uz")
# plot!(1:iterations, residuals.p, label="p")
