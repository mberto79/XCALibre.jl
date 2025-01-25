using XCALibre
# using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

# mesh_dev = adapt(CUDABackend(), mesh)
mesh_dev = mesh

Umag = 0.5
velocity = [Umag, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

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
    Wall(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

 @assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(time=Euler),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # DILU(), TEMPORARY!
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    ),
    p = set_solver(
        model.momentum.p;
        solver      = GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #LDL(), TEMPORARY!
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    )
)

runtime = set_runtime(
    iterations=1000, time_step=0.005, write_interval=1000)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing

residuals = run!(model, config);

inlet = boundary_average(:inlet, model.momentum.U, config)
outlet = boundary_average(:outlet, model.momentum.U, config)

@test outlet ≈ 0.5*inlet atol=0.1*Umag