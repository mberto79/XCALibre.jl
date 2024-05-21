using Plots
using FVM_1D
using Krylov
using CUDA
using KernelAbstractions


# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh_file = "unv_sample_meshes/cylinder_d10mm_2mm.unv"
mesh_file = "unv_sample_meshes/cylinder_d10mm_10-7.5-2mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh, integer=Int32, float=Float32)
mesh = update_mesh_format(mesh)

# Inlet conditions

velocity = [0.50, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-5
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #NormDiagonal(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-5
    )
)

schemes = (
    U = set_schemes(time=Euler, divergence=Linear, gradient=Midpoint),
    p = set_schemes(time=Euler, divergence=Linear, gradient=Midpoint)
)


runtime = set_runtime(
    iterations=10000, write_interval=50, time_step=0.005)
    # iterations=1000, write_interval=50, time_step=0.005)

# 2mm mesh use settings below (to lower Courant number)
runtime = set_runtime(
        iterations=5000, write_interval=250, time_step=0.001) # Only runs on 32 bit

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Rx, Ry, Rz, Rp, model = piso!(model, config); #, pref=0.0)

plot(; xlims=(0,runtime.iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")