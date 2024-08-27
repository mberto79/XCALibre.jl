using Plots
using FVM_1D
using CUDA
using Adapt


mesh_file = "unv_sample_meshes/UNV_BFS_3D_periodic_abhi.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

backend = CUDABackend()
hardware = set_hardware(backend=backend, workgroup=32)


periodicPair = construct_periodic(mesh, backend, :side1, :side2)
mesh_dev = adapt(backend, mesh)

# INLET CONDITIONS 

Umag = 0.05 #7.65
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1.5e-5
L = 0.1
νR = 5
Tu = 0.01
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
Re = (L*velocity[1])/nu
δt = 10e-3
δx = 1e-3
Co = Umag*δt/δx
Pex = Umag*δx/nu

# Steady configuration 

model= Physics(
    time = Steady(),
    fluid = FLUID{Incompressible}(nu = nu),
    # turbulence = LES{Smagorinsky}(),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, noSlip),
    Symmetry(:top, 0.0),
    Symmetry(:side1, 0.0),
    Symmetry(:side2, 0.0),
    # periodicPair...
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0),
    # periodicPair...
)

@assign! model turbulence nut (
    # Neumann(:inlet, 0.0),
    # Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    # Neumann(:top, 0.0), 
    # Neumann(:side1, 0.0), 
    # Neumann(:side2, 0.0), 
    # periodicPair...
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.5,
        rtol = 1e-1,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2,
        atol = 1e-10
    ),
)

runtime = set_runtime(
    iterations=1000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.nut, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config)

# Set up transient run

modelTransient = change(model, :time, Transient())

schemes = (
    U = set_schemes(divergence=Linear, gradient=Midpoint, time=Euler),
    p = set_schemes(gradient=Midpoint, time=Euler)
)

solvers = (
    U = set_solver(
        modelTransient.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0,
        atol = 1e-5
    ),
    p = set_solver(
        modelTransient.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 0,
        atol = 1e-6
    ),
)

runtime = set_runtime(
    iterations=1000, write_interval=100, time_step=δt)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

Rx, Ry, Rz, Rp, model_out = run!(modelTransient, config)

