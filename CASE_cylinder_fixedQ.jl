using Plots
using FVM_1D
using Krylov


# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

# Inlet conditions

velocity = [0.50, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
pressure = 100000.0
h_inf = 300*1005 
h_wall = 320*1005 
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = RANS{Laminar_rho}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    # Wall(:cylinder, noSlip),
    Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, pressure),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy ( 
    Dirichlet(:inlet, h_inf),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, h_wall),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol        = 1e-3,
        atol        = 1e-6,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol        = 1e-3,
        atol        = 1e-6,
    ),
    energy = set_solver(
        model.energy;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol        = 1e-3,
        atol        = 1e-6,
    ),
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Upwind, gradient=Midpoint),
    energy = set_schemes(divergence=Upwind)
    # U = set_schemes(divergence=Linear),
    # p = set_schemes(divergence=Linear, gradient=Midpoint),
    # energy = set_schemes(divergence=Linear)
)

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, pressure)
initialise!(model.energy, h_inf)

Rx, Ry, Rz, Rp, Re = simple_rho_K!(model, config)

plot(; xlims=(0,runtime.iterations))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Re), Re, yscale=:log10, label="energy")