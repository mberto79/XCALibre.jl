using Plots
using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
# grid = "cylinder_d10mm_2mm.unv"
# grid = "cylinder_d10mm_10-7.5-2mm.unv"
mesh_file = joinpath(grids_dir, grid)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = set_hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Inlet conditions

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-2
Re = (0.2*velocity[1])/nu
gamma = 1.4
cp = 1005.0
temp = 300.0
pressure = 100000
Pr = 0.7

model = Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=288.15),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Neumann(:outlet, 0.0),
            Wall(:cylinder, noSlip),
            Symmetry(:bottom, 0.0),
            Symmetry(:top, 0.0)
        ],
        p = [
            Neumann(:inlet, 0.0),
            Dirichlet(:outlet, pressure),
            Wall(:cylinder, 0.0),
            Neumann(:bottom, 0.0),
            Neumann(:top, 0.0)
        ],
        h = [
            FixedTemperature(:inlet, T=300.0, model=model.energy),
            Neumann(:outlet, 0.0),
            FixedTemperature(:cylinder, T=330.0, model=model.energy),
            Neumann(:bottom, 0.0),
            Neumann(:top, 0.0)
        ]
    )
)

solvers = (
    U = set_solver(
        region = mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1,
    ),
    p = set_solver(
        region = mesh_dev,
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2
    ),
    h = set_solver(
        region = mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1
    )
)

schemes = (
    U = set_schemes(divergence=LUST, gradient=Midpoint),
    p = set_schemes(divergence=LUST, gradient=Midpoint),
    h = set_schemes(divergence=LUST, gradient=Midpoint)
)

runtime = set_runtime(iterations=500, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)

println("Maxh ", maximum(model.energy.T.values), " minh ", minimum(model.energy.T.values))

residuals = run!(model, config, ncorrectors=2); #, pref=0.0)

plot(; xlims=(0,runtime.iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Rh), Rh, yscale=:log10, label="h")