using Plots
using FVM_1D
using Krylov
using KernelAbstractions
using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
# mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh_file = "unv_sample_meshes/backwardFacingStep_5mm.unv"
mesh_file = "unv_sample_meshes/backwardFacingStep_2mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

nu = 1e-3
# u_mag = 3.5 # 2mm mesh
u_mag = 1.5 # 5mm mesh
velocity = [u_mag, 0.0, 0.0]
k_inlet = 1
ω_inlet = 1000
ω_wall = ω_inlet
Re = velocity[1]*0.1/nu

model = RANS{KOmega}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    OmegaWallFunction(:top)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0), 
    Dirichlet(:top, 0.0)
)

schemes = (
    U = set_schemes(gradient=Midpoint),
    p = set_schemes(gradient=Midpoint),
    k = set_schemes(gradient=Midpoint),
    omega = set_schemes(gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-5,
        atol = 1e-2
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-5,
        atol = 1e-3
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-5,
        atol = 1e-2
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-5,
        atol = 1e-2
    )
)

runtime = set_runtime(iterations=2000, write_interval=100, time_step=1)
runtime = set_runtime(iterations=1, write_interval=-1, time_step=1)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

Rx, Ry, Rz, Rp, model = simple!(model, config) # 36.90k allocs

Reff = stress_tensor(model.U, nu, model.turbulence.nut)
Fp = pressure_force(:wall, model.p, 1.25)
Fv = viscous_force(:wall, model.U, 1.25, nu, model.turbulence.nut)

plot(; xlims=(0,494))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")