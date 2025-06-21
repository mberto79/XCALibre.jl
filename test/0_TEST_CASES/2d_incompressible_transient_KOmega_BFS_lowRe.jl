using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_5mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

nu = 1e-3
Umag = 1.5
velocity = [Umag, 0.0, 0.0]
k_inlet = 1
ω_inlet = 1000
ω_wall = ω_inlet
Re = velocity[1]*0.1/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Neumann(:outlet, 0.0),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Dirichlet(:top, [0.0, 0.0, 0.0])
        ],
        p = [
            Neumann(:inlet, 0.0),
            Dirichlet(:outlet, 0.0),
            Neumann(:wall, 0.0),
            Neumann(:top, 0.0)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Neumann(:outlet, 0.0),
            Dirichlet(:wall, 0.0),
            Dirichlet(:top, 0.0)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Neumann(:outlet, 0.0),
            OmegaWallFunction(:wall),
            OmegaWallFunction(:top)
        ],
        nut = [
            Dirichlet(:inlet, k_inlet/ω_inlet),
            Neumann(:outlet, 0.0),
            Dirichlet(:wall, 0.0), 
            Dirichlet(:top, 0.0)
        ],
    )
)

schemes = (
    U = set_schemes(gradient=Midpoint, time=Euler),
    p = set_schemes(gradient=Midpoint),
    k = set_schemes(gradient=Midpoint, time=Euler),
    omega = set_schemes(gradient=Midpoint, time=Euler)
)

solvers = (
    U = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    ),
    p = set_solver(
        region=mesh_dev,
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    ),
    k = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    ),
    omega = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    )
)

runtime = set_runtime(
    iterations=500, write_interval=500, time_step=0.01)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=backend, workgroup=workgroup)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing
@test initialise!(model.turbulence.k, k_inlet) === nothing
@test initialise!(model.turbulence.omega, ω_inlet) === nothing
@test initialise!(model.turbulence.nut, k_inlet/ω_inlet) === nothing

residuals = run!(model, config);

inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)

@test outlet ≈ 0.5*inlet atol=0.1*Umag