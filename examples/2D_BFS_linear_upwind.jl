using XCALibre
# using CUDA # uncomment to run on an NVIDIA GPU

# Laminar backward-facing step with second-order upwind convection for U. `LinearUpwind`
# reconstructs face values from the upwind-cell gradient (here limited with `CellBased`);
# `LimitedLinearUpwind` also bounds them between neighbouring cell values.
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "backwardFacingStep_10mm.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)
# backend = CUDABackend(); workgroup = 32

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [1.5, 0.0, 0.0]
nu = 1e-3

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
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Symmetry(:top)
        ]
    )
)

schemes = (
    U = Schemes(divergence=LinearUpwind, limiter=CellBased()), # or LimitedLinearUpwind
    p = Schemes()
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-2
    ),
    p = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-3
    )
)

runtime = Runtime(iterations=2000, time_step=1, write_interval=2000)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)
