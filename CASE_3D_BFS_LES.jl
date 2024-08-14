using Plots
using FVM_1D
using CUDA


mesh_file = "unv_sample_meshes/bfs_unv_hex_5mm.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

mesh_gpu = adapt(CUDABackend(), mesh)

# INLET CONDITIONS 

Umag = 3.5
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-5
νR = 5
Tu = 0.01
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Transient(),
    fluid = FLUID{Incompressible}(nu = nu),
    turbulence = LES{Smagorinsky}(),
    energy = nothing,
    domain = mesh
    )

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, noSlip),
    Neumann(:sides, 0.0),
    Neumann(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:sides, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:sides, 0.0), 
    Neumann(:top, 0.0), 
    Dirichlet(:wall, 0.0)
)

schemes = (
    U = set_schemes(divergence=Linear, gradient=Midpoint, time=Euler),
    p = set_schemes(gradient=Midpoint, time=Euler)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-5
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-5
    ),
)

runtime = set_runtime(
    iterations=10000, write_interval=200, time_step=0.0001)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

Rx, Ry, Rz, Rp, model = run!(model, config); #, pref=0.0)

Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
Fp = pressure_force(:cylinder, model.momentum.p, 1.25)
Fv = viscous_force(:cylinder, model.momentum.U, 1.25, nu, model.turbulence.nut)

plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")