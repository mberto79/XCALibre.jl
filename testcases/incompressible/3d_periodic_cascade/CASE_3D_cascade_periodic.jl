using XCALibre
using Adapt
# using CUDA

mesh_file = "testcases/incompressible/3d_periodic_cascade/cascade_3D_periodic_2p5mm.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend() # Uncomment this if using GPU
backend = CPU() # Uncomment this if using CPU
periodic = construct_periodic(mesh, backend, :top, :bottom)
# mesh_dev = adapt(CUDABackend(), mesh)  # Uncomment this if using GPU

velocity = [0.25, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh # mesh_dev  # use mesh_dev for GPU backend
    )


    
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:plate, [0.0, 0.0, 0.0]),
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
    U = set_schemes(divergence=Linear, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, #CgSolver, # BicgstabSolver, GmresSolver, #CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2,
        atol = 1e-10
    )
)

runtime = set_runtime(
    iterations=500, time_step=1, write_interval=100)

hardware = set_hardware(backend=CPU(), workgroup=4)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")