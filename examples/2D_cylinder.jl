using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# using ThreadedSparseCSR 
# ThreadedSparseCSR.multithread_matmul(BaseThreads())

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Inlet conditions

velocity = [0.2, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Wall(:cylinder, noSlip),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ],
        p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:cylinder),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 0.1
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 0.01
    )
)

grad = Gauss # Midpoint # Gauss
schemes = (
    U = Schemes(divergence=Linear, gradient=grad),
    p = Schemes(gradient=grad)
)

# runtime = Runtime(iterations=20, write_interval=10, time_step=1) # for proto
runtime = Runtime(iterations=1000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config, ncorrectors=0)

# xrange = 1:runtime.iterations
# plot(; xlims=(0,runtime.iterations), ylims=(1e-7,0.2))
# plot!(xrange, residuals.Ux, yscale=:log10, label="Ux")
# plot!(xrange, residuals.Uy, yscale=:log10, label="Uy")
# plot!(xrange, residuals.p, yscale=:log10, label="p")