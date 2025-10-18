using XCALibre
# using CUDA # uncomment to run on GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Inlet conditions
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Transient(),
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
                Wall(:cylinder, noSlip),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ],
        p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:cylinder),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-5
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Jacobi(), #NormDiagonal(), # DILU()
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-4,
        atol = 1e-5
    )
)

# timeScheme = Euler
timeScheme = CrankNicolson
schemes = (
    U = Schemes(time=timeScheme, divergence=LUST, gradient=Gauss),
    p = Schemes(time=timeScheme, gradient=Gauss)
)


runtime = Runtime(
    iterations=1000, write_interval=50, time_step=0.005) # uncomment to save files
    # iterations=1000, write_interval=-1, time_step=0.005) # used to run only

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)