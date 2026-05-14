using XCALibre
using LinearAlgebra

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

runtime = Runtime(iterations=600, write_interval=600, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

MRFCell = 2000 
inertialCell = 10

@test rotating_frames.global_mask[MRFCell] == 1.0
@test rotating_frames.global_mask[inertialCell] == 0.0

U = model.momentum.U
mesh = U.mesh
cells = mesh.cells
x0 = rotating_frames.frames.x0[1]
rotaxis=rotating_frames.frames.rotaxis[1]

Up = VectorField(mesh)

for i ∈ eachindex(Up.x.values)
    r = cells[i].centre - x0
    r_norm = r./norm(r)
    tang = r_norm × rotaxis
    Up.x.values[i] = U[i] ⋅ r_norm
    Up.z.values[i] = U.z.values[i]
    Up.y.values[i] = -(U[i] ⋅ tang)
end

maxval = maximum(Up.x.values)

@test maxval < 0.5       # In a perfectly converged case, the outwards velocity should be 0 everywhere. At 600 
                         # iterations the max should always be below 0.5 as it heads towards 0 with more iterations