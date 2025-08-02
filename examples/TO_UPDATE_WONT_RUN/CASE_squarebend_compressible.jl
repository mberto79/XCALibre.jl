using Plots
using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/OF_squareBend_laminar/constant/polyMesh/"
mesh = FOAM3D_mesh(mesh_file, scale=1.0, integer_type=Int64, float_type=Float64)


# mesh_dev = adapt(CUDABackend(), mesh)

# Inlet conditions

# Not working

velocity = [400, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu
gamma = 1.4
cp = 1005.0
temp = 300.0
pressure = 110000.0
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
    energy = Energy{SensibleEnthalpy}(Tref = 298.15),
    domain = mesh
    )

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:walls, noSlip)
)

@assign! model momentum p (
    Dirichlet(:inlet, pressure),
    Neumann(:outlet, 0.0),
    Neumann(:walls, 0.0)
)

@assign! model energy h (
    FixedTemperature(:inlet, T=300.0, Enthalpy(cp=cp, Tref=298.15)),
    Neumann(:outlet, 0.0),
    Neumann(:walls, 0.0)
)

solvers = (
    U = SolverSetup(
        model.momentum.U;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-4
    ),
    p = SolverSetup(
        model.momentum.p;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        limit = (1000.0, 1000000.0),
        rtol = 1e-2,
        atol = 1e-4
    ),
    h = SolverSetup(
        model.energy.h;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        # limit = (100.0, 1000.0),
        rtol = 1e-2,
        atol = 1e-4
    )
)

schemes = (
    U = Schemes(divergence=BoundedUpwind),#, gradient=Midpoint),
    p = Schemes(divergence=Upwind, gradient=Midpoint),
    h = Schemes(divergence=BoundedUpwind)#, gradient=Midpoint)
)

runtime = Runtime(iterations=5, write_interval=1, time_step=1)

hardware = Hardware(backend=CPU(), workgroup=4)
# hardware = Hardware(backend=CUDABackend(), workgroup=32)
# hardware = Hardware(backend=ROCBackend(), workgroup=32)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, [0.0, 0.0, 0.0])
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)

residuals = run!(model); #, pref=0.0)

plot(; xlims=(0,runtime.iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Rh), Rh, yscale=:log10, label="h")