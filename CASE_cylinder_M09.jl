using Plots
using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_2mm.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_10-7.5-2mm.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_gpu = adapt(CUDABackend(), mesh)

# Inlet conditions

velocity = [300, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 0.0#1e-3
Re = (0.2*velocity[1])/nu
gamma = 1.4
cp = 1005.0
temp = 300.0
pressure = 100000.0
Pr = 0.7

model = Physics(
    time = Steady(),
    fluid = Compressible(
        mu = nu,
        cp = ConstantScalar(cp),
        gamma = ConstantScalar(gamma),
        Pr = ConstantScalar(Pr)
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(),
    domain = mesh
    )

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    # Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, pressure),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy h (
    FixedTemperature(:inlet, T=300.0, model=model.energy),
    Neumann(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    # FixedTemperature(:cylinder, T=330.0, model=model.energy),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-4
    ),
    p = set_solver(
        model.momentum.p;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-4
    ),
    h = set_solver(
        model.energy.h;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-4
    )
)

schemes = (
    U = set_schemes(divergence=Upwind),#, gradient=Midpoint),
    p = set_schemes(divergence=Linear, gradient=Midpoint),
    h = set_schemes(divergence=Upwind)#, gradient=Midpoint)
)

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)

hardware = set_hardware(backend=CPU(), workgroup=4)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)

println("Maxh ", maximum(model.energy.T.values), " minh ", minimum(model.energy.T.values))

Rx, Ry, Rz, Rp, Rh, model = run!(model, config); #, pref=0.0)

plot(; xlims=(0,runtime.iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Rh), Rh, yscale=:log10, label="h")