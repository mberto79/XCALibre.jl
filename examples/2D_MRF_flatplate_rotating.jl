using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "spinning_rod_mesh_V3.unv"
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

rotating_frames = RotatingFrames2D(          #  "rotating_frames = RotatingFrames3D("  for 3D meshes.
    hardware=hardware,
    mesh=mesh,
    frames = [
        RotatingFrame(
            omega = 15,
            x1 = [0.0, 0.0, 1.0],
            x0 = [0.0, 0.0, 0.0],
            radius_inner = 0.0,
            radius_outer = 0.05,
            hardware=hardware,
            mesh=mesh
        )                                # You can define multiple rotating regios, given they do not overlap and have at least 1 cell between them.
#       RotatingFrame(
#            omega = 25,
#            rotaxis = [0.0, 0.0, 1.0],
#            x0 = [-0.1, 0.0, 0.0],
#            radius_inner = 0.0,
#            radius_outer = 0.05,
#            hardware=hardware,
#            mesh=mesh
#       )
    ]
)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev,
    reference_frames = rotating_frames
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
#            RotatingWall(                       # Each rotating frame needs a respective rotating wall BC for all rotating walls inside.
#                :rotor2,
#                rpm=(30/pi)*rotating_frames.frames.omega[2],
#                centre=rotating_frames.frames.x0[2],
#                axis=rotating_frames.frames.rotaxis[2]
#                ),
            Wall(:walls, [0.0, 0.0, 0.0])
        ],
        p = [
            Wall(:rotor),
            Wall(:walls)
        ],
        k = [
            KWallFunction(:rotor)
            KWallFunction(:walls)
        ],
        omega = [
            OmegaWallFunction(:rotor)
            OmegaWallFunction(:walls)
        ],
        nut = [
            NutWallFunction(:rotor)
            NutWallFunction(:walls)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Upwind),
    k = Schemes(divergence=Upwind),
    omega = Schemes(divergence=Upwind)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.5,
        rtol = 1e-2,
        atol = 1e-10
    ),
    p = SolverSetup(
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.5,
        rtol = 1e-3,
        atol = 1e-10
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 5e-7,
        relax       = 0.5,
        rtol = 1e-2,
        atol = 1e-10
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.5,
        rtol = 1e-2,
        atol = 1e-10
    )
)

runtime = Runtime(iterations=100, write_interval=10, time_step=1)

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
mesh_name = get_mesh_name(mesh_file)
velocity_name = string("velocity_",u_mag)*'_'
omega_name = string("omega_",omega)*'_'
script_name = string(@__FILE__)
output_dir = mesh_name * omega_name *  "polarCoords_MRFdemo"
pattern = "vtk"
# pattern = "foam"
output_directory(output_dir, script_name)

#using Plots
#iterations = runtime.iterations
#plot(yscale=:log10, ylims=(1e-8,1e-1))
#plot!(1:iterations, residuals.Ux, label="Ux")
#plot!(1:iterations, residuals.Uy, label="Uy")
#plot!(1:iterations, residuals.Uz, label="Uz")
#plot!(1:iterations, residuals.p, label="p")
#plot!(size=(800,600))