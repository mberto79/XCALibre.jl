using XCALibre
# using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid = "cascade_3D_periodic_2p5mm.unv"
grid = "cascade_3D_periodic_4mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV3D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh3

# backend = CUDABackend() # Uncomment this if using GPU
backend = CPU() # Uncomment this if using CPU
workgroup = workgroupsize(mesh)
periodic = construct_periodic(mesh, backend, :top, :bottom)
mesh_dev = adapt(backend, mesh)

velocity = [0.25, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev # mesh_dev  # use mesh_dev for GPU backend
    )

BCs = assign(
    region=mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:plate, [0.0, 0.0, 0.0]),
            Extrapolated(:side1),
            Extrapolated(:side2),
            periodic...
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            Extrapolated(:plate),
            Extrapolated(:side1),
            Extrapolated(:side2),
            periodic...
        ]
    )
)

schemes = (
    U = Schemes(divergence=Linear, gradient=Midpoint),
    p = Schemes(gradient=Midpoint)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), #Cg(), # Bicgstab(), Gmres(), #Cg()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2
    )
)

runtime = Runtime(
    iterations=100, time_step=1, write_interval=100)

hardware = Hardware(backend=backend, workgroup=workgroup)
# hardware = Hardware(backend=CUDABackend(), workgroup=32)
# hardware = Hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing

residuals = run!(model, config)

# test periodic boundaries agree (velocity)
top = boundary_average(:top, model.momentum.U, BCs.U, config)
bottom = boundary_average(:bottom, model.momentum.U, BCs.U, config)

@test top ≈ bottom

# test periodic boundaries agree (pressure)
top = boundary_average(:top, model.momentum.p, BCs.p, config)
bottom = boundary_average(:bottom, model.momentum.p, BCs.p, config)

@test top ≈ bottom



