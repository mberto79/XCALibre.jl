using Plots
using FVM_1D
using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_2mm.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_10-7.5-2mm.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_gpu = adapt(CUDABackend(), mesh)
mesh_gpu = mesh

# Inlet conditions

velocity = [0.2, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Steady(),
    fluid = FLUID{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{Isothermal}(),
    domain = mesh_gpu
    )

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
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
    )
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(divergence=Upwind, gradient=Midpoint)
)

# runtime = set_runtime(iterations=20, write_interval=10, time_step=1) # for proto
runtime = set_runtime(iterations=100, write_interval=100, time_step=1)

hardware = set_hardware(backend=CPU(), workgroup=4)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config)

plot(; xlims=(0,runtime.iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")