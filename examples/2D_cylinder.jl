using XCALibre
using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# using ThreadedSparseCSR 
# ThreadedSparseCSR.multithread_matmul(BaseThreads())

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = mesh
mesh_dev = adapt(CUDABackend(), mesh)

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
        rtol = 0.1,
        atol = 1e-6
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 0.1,
        atol = 1e-6
    )
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Orthogonal),
    p = set_schemes(gradient=Orthogonal)
)

# runtime = set_runtime(iterations=20, write_interval=10, time_step=1) # for proto
runtime = set_runtime(iterations=500, write_interval=100, time_step=1)

hardware = set_hardware(backend=CPU(), workgroup=1024)
hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

plot(; xlims=(0,runtime.iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

using CUDA
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR

n = 5
A = sprand(n,n, 0.5)

i, j, v = findnz(A)

Acsc = sparse(i, j, v, n ,n)
Acsr = sparsecsr(i, j, v, n ,n)

Agpu = CUSPARSE.CuSparseMatrixCSR(Acsc)

i, j, v = findnz(A) |> cu

Agpu = CUSPARSE.CuSparseMatrixCSR(i, j, v, (1000, 1000))