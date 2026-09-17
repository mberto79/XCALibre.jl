using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "3d_streamtube_1.0x0.1x0.1_0.08mm.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV3D_mesh(mesh_file, scale=1.0)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [1.0, 0.0, 0.0]
nu = 1e-3

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

sides = [:top, :bottom, :side1, :side2]

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Slip.(sides)...
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Slip.(sides)...
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(gradient=Gauss)
)

solvers = (
    U = SolverSetup(
        solver = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.7
    ),
    p = SolverSetup(
        solver = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.3
    )
)

runtime = Runtime(iterations=100, time_step=1, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime,
    hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# Project the initial guess onto a divergence-free potential-flow field.
# The potential boundary conditions are inferred from those assigned to p.
result = potential_flow!(model, config; ncorrectors=5)
println("potential flow residual: ", result.residual)

residuals = run!(model, config)
