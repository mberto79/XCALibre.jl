using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "SpinningSquare0p2Diameter.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

nu = 1e-3
u_mag = 0.0
velocity = [u_mag, 0.0, 0.0]
Tu = 0.05
nuR = 100
k_inlet = 1 #3/2*(Tu*u_mag)^2
ω_inlet = 1000 #k_inlet/(nuR*nu)
νt_inlet = k_inlet/ω_inlet
Re = velocity[1]*0.1/nu

rotating_frames = RotatingFrames2D(  #  "rotating_frames = RotatingFrames3D("  for 3D meshes.
    hardware=hardware,
    mesh=mesh,
    frames = [
        RotatingFrame(
            omega = 5,
            x1 = [0.0, 0.0, 1.0],
            x0 = [0.0, 0.0, 0.0],
            radius_inner = 0.0,
            radius_outer = 0.1,
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
                :walls,
                rpm=(30/pi)*rotating_frames.frames.omega[1],
                centre=rotating_frames.frames.x0[1],
                axis=rotating_frames.frames.rotaxis[1]
                ),
            Wall(:boundary, [0.0, 0.0, 0.0])
        ],
        p = [
            Wall(:boundary),
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
        convergence = 1e-5,
        relax       = 0.3,
        rtol = 1e-3,
        atol = 1e-10
    )
)

runtime = Runtime(iterations=10, write_interval=1, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, νt_inlet)

residuals = run!(model, config)

# Custom Output Functions
mesh_name = string(get_mesh_name(mesh_file))*'_'
omega_name = string("omega_",omega)*'_'
script_name = string(@__FILE__)*'_'
output_dir = script_name * mesh_name * omega_name *  "_MRFdemo"
output_directory(output_dir, script_name)