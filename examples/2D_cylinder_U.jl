using XCALibre
# using CUDA # uncomment to run on GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = mesh
# mesh_dev = adapt(CUDABackend(), mesh) # uncomment to run on GPU

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

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Wall(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3,
        # atol = 1e-15
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #NormDiagonal(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-4,
        # atol = 1e-15
    )
)

schemes = (
    U = set_schemes(time=Euler, divergence=LUST, gradient=Midpoint),
    p = set_schemes(time=Euler, gradient=Midpoint)
)


runtime = set_runtime(
    iterations=1000, write_interval=50, time_step=0.005) # uncomment to save files
    # iterations=1000, write_interval=-1, time_step=0.005) # used to run only

hardware = set_hardware(backend=CPU(), workgroup=cld(length(mesh.cells), Threads.nthreads()))
# hardware = set_hardware(backend=CUDABackend(), workgroup=32) # uncomment to run on GPU

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)