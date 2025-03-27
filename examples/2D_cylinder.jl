using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# using ThreadedSparseCSR 
# ThreadedSparseCSR.multithread_matmul(BaseThreads())

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend= CPU(); workgroup = 1024
# backend= CUDABackend(); workgroup = 32
hardware = set_hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(hardware.backend, mesh)

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

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:cylinder, noSlip),
    # Neumann.([:top, :bottom], [0.0, 0.0])...,
    Symmetry.([:top, :bottom])...,
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    # Neumann.([:top, :bottom], [0.0, 0.0])...,
    Symmetry.([:top, :bottom])...,
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 0.1
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 0.01
    )
)

grad = Orthogonal # Midpoint # Orthogonal
schemes = (
    U = set_schemes(divergence=Linear, gradient=grad),
    p = set_schemes(gradient=grad)
)

# runtime = set_runtime(iterations=20, write_interval=10, time_step=1) # for proto
runtime = set_runtime(iterations=500, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config, ncorrectors=0)

xrange = 1:runtime.iterations
plot(; xlims=(0,runtime.iterations), ylims=(1e-7,0.2))
plot!(xrange, residuals.Ux, yscale=:log10, label="Ux")
plot!(xrange, residuals.Uy, yscale=:log10, label="Uy")
plot!(xrange, residuals.p, yscale=:log10, label="p")