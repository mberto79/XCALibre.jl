using Plots
using FVM_1D
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "testcases/incompressible/2d_backwards_step/backwardFacingStep_10mm.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)
# mesh_gpu = adapt(CUDABackend(), mesh)  # Uncomment this if using GPU

# Inlet conditions
velocity = [1.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh # mesh_gpu  # use mesh_gpu for GPU backend
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Symmetry(:wall, [0.0, 0.0, 0.0]),
    Symmetry(:top, [0.0, 0.0, 0.0]),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence = Linear),
    p = set_schemes()
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    )
)

runtime = set_runtime(
    iterations=2000, time_step=1, write_interval=100)

hardware = set_hardware(backend=CPU(), workgroup=4)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config)