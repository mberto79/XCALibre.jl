using Plots
using FVM_1D
using Krylov
using KernelAbstractions
using CUDA

mesh_file = "unv_sample_meshes/backwardFacingStep_5mm_long.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_gpu = adapt(CUDABackend(), mesh)

nu = 1e-3
u_mag = 3.5
velocity = [u_mag, 0.0, 0.0]
Tu = 0.05
nuR = 50
k_inlet = 3/2*(Tu*u_mag)^2
ω_inlet = k_inlet/(nuR*nu)
νt_inlet = k_inlet/ω_inlet
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Incompressible(nu = ConstantScalar(nu)),
    turbulence = RANS{KOmega}(),
    energy = nothing,
    domain = mesh_gpu
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    KWallFunction(:wall),
    KWallFunction(:top)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    OmegaWallFunction(:top)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, νt_inlet),
    Neumann(:outlet, 0.0),
    NutWallFunction(:wall), 
    NutWallFunction(:top)
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Upwind),
    k = set_schemes(divergence=Upwind),
    omega = set_schemes(divergence=Upwind)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-15,
        atol = 1e-5
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-15,
        atol = 1e-5
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-15,
        atol = 1e-5
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-15,
        atol = 1e-5
    )
)

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)
# runtime = set_runtime(iterations=2, write_interval=-1, time_step=1)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, νt_inlet)

Rx, Ry, Rz, Rp, model_out = run!(model, config) # 36.90k allocs

Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
Fp = pressure_force(:wall, model.momentum.p, 1.25)
Fv = viscous_force(:wall, model.momentum.U, 1.25, nu, model.turbulence.nut)


plot(; xlims=(0,494))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# PROFILING CODE

using Profile, PProf

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, νt_inlet)

Rx, Ry, Rz, Rp, model_out = run!(model, config)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=0.1 begin 
Rx, Ry, Rz, Rp, model_out = run!(model, config)
end

PProf.Allocs.pprof()