using XCALibre
using CUDA


mesh_file = "unv_sample_meshes/UNV_BFS_3D_periodic_abhi.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

backend = CUDABackend()
# periodic1, periodic2 = construct_periodic(mesh, backend, :side1, :side2)
periodic, connectivity, mesh= construct_periodic(mesh, backend, :side1, :side2)
mesh_dev = adapt(backend, mesh)

# INLET CONDITIONS 

Umag = 0.5
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
νR = 5
Tu = 0.01
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
Re = (0.1*velocity[1])/nu

model = Physics(
    time = Transient(),
    fluid = FLUID{Incompressible}(nu = nu),
    # turbulence = LES{Smagorinsky}(),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{Isothermal}(),
    domain = mesh_dev,
    periodic = connectivity
    )

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, noSlip),
    # Symmetry(:top, 0.0),
    Dirichlet(:top, velocity),
    # Symmetry(:side1, 0.0),
    # Symmetry(:side2, 0.0),
    periodic...
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0),
    # Neumann(:side1, 0.0),
    # Neumann(:side2, 0.0),
    periodic...
)

@assign! model turbulence nut (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Neumann(:top, 0.0), 
    Neumann(:side1, 0.0), 
    Neumann(:side2, 0.0), 
    # periodic...
)

schemes = (
    U = set_schemes(divergence=Linear, gradient=Midpoint, time=Euler),
    p = set_schemes(gradient=Midpoint, time=Euler)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-5
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-6
    ),
)

runtime = set_runtime(
    iterations=2000, write_interval=50, time_step=1e-8)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.nut, nu)

Rx, Ry, Rz, Rp, model_out = run!(model, config); #, pref=0.0)
