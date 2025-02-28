using XCALibre
# using Adapt
# using CUDA

mesh_file = "unv_sample_meshes/cascade_3D_periodic_2p5mm.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend()
backend = CPU()
workgroup = cld(length(mesh.cells), Threads.nthreads())
# sidePeriodic, sideConnectivity = construct_periodic(mesh, backend, :side1, :side2)
periodic, connectivity, mesh_periodic = construct_periodic(mesh, backend, :top, :bottom)

# mesh_dev = adapt(CUDABackend(), mesh_periodic)
mesh_dev = mesh_periodic

velocity = [0.25, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = FLUID{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{Isothermal}(),
    domain = mesh_dev,
    periodic = connectivity
    )


    
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:plate, [0.0, 0.0, 0.0]),
    # Symmetry(:side1, 0.0),
    # Symmetry(:side2, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0),
    periodic...
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:plate, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0),
    periodic...
)

schemes = (
    # U = set_schemes(divergence=Upwind, gradient=Midpoint),
    U = set_schemes(divergence=Linear, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint)
    # p = set_schemes()
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, #CgSolver, # BicgstabSolver, GmresSolver, #CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-2,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-3,
        atol = 1e-10
    )
)

runtime = set_runtime(
    iterations=500, time_step=1, write_interval=100)

hardware = set_hardware(backend=backend, workgroup=workgroup)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
# initialise!(model.momentum.U, [0.0, 0.0, 0.0 ])
initialise!(model.momentum.p, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config)