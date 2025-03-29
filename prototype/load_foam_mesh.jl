using XCALibre

mesh_file = "constant/polyMesh"
mesh = FOAM3D_mesh(mesh_file)


# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024
hardware = set_hardware(backend=backend, workgroup=workgroup)

mesh_dev = adapt(hardware.backend, mesh)

# Inlet conditions
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-5
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
    Wall(:bottom, noSlip),
    Neumann(:top, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Wall(:bottom, 0.0),
    Neumann(:top, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0)
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
        rtol = 0.01,
        itmax = 1000
    )
)

schemes = (
    U = set_schemes(time=SteadyState, divergence=Upwind, gradient=Midpoint),
    p = set_schemes(time=SteadyState, gradient=Midpoint)
)


GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

# Now get timing information

runtime = set_runtime(iterations=100, write_interval=10, time_step=1)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

@time residuals = run!(model, config, output=OpenFOAM())

# iterations = runtime.iterations
# plot(yscale=:log10, ylims=(1e-7,1e-1))
# plot!(1:iterations, residuals.Ux, label="Ux")
# plot!(1:iterations, residuals.Uy, label="Uy")
# plot!(1:iterations, residuals.Uz, label="Uz")
# plot!(1:iterations, residuals.p, label="p")