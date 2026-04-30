using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "rotating_flatPlate.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

nu = 1e-3
u_mag = 0.0
velocity = [u_mag, 0.0, 0.0]

rotating_frames = RotatingFrames2D(
    hardware=hardware,
    mesh=mesh,
    frames = [
        RotatingFrame(
            omega = 10,
            x1 = [0.0, 0.0, 1.0],
            x0 = [0.0, 0.0, 0.0],
            radius_inner = 0.0,
            radius_outer = 0.2,
            hardware=hardware,
            mesh=mesh
        )
    ],
    polar=true
)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible_MRF}(nu = nu, refFrames = rotating_frames),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

BCs = assign(
    region = mesh_dev,
    (
        U = [
            RotatingWall(
                :rotor,
                rpm=(30/pi)*rotating_frames.frames.omega[1],
                centre=rotating_frames.frames.x0[1],
                axis=rotating_frames.frames.rotaxis[1]
                ),
            Wall(:walls, [0.0, 0.0, 0.0])
        ],
        p = [
            Wall(:rotor),
            Wall(:walls)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Upwind)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-6,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-10
    ),
    p = SolverSetup(
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-6,
        relax       = 0.3,
        rtol = 1e-3,
        atol = 1e-10
    )
)

runtime = Runtime(iterations=500, write_interval=500, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

MRFCell = 10
inertialCell = 10000

@test rotating_frames.global_mask[MRFCell] = 1.0
@test rotating_frames.global_mask[inertialCell] = 0.0