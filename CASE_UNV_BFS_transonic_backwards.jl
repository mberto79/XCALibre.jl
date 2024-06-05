using Plots
using FVM_1D
using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
# mesh = update_mesh_format(mesh; integer=Int32, float=Float32)
mesh = update_mesh_format(mesh)

velocity = [-150, 0.0, 0.0]
nu = 0.001
Re = velocity[1]*0.1/nu
Cp = 1005
gamma = 1.4
h_inf = 300*Cp
h_wall = 320*Cp
pressure = 100000

model = RANS{Laminar_rho}(mesh=mesh, viscosity=ConstantScalar(nu))
thermodel = ThermoModel{IdealGas}(mesh=mesh, Cp=Cp, gamma=gamma)

@assign! model U (
    Dirichlet(:outlet, velocity),
    Neumann(:inlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),#Wall(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])#Wall(:top, [0.0, 0.0, 0.0])
)

 @assign! model p (
    Dirichlet(:outlet, pressure),
    Neumann(:inlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy ( 
    Dirichlet(:outlet, h_inf),
    Neumann(:inlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Linear, gradient=Midpoint),
    energy = set_schemes(divergence=Upwind)
    # U = set_schemes(divergence=Linear),
    # p = set_schemes(divergence=Linear, gradient=Midpoint),
    # energy = set_schemes(divergence=Linear)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol        = 1e-2,
        atol        = 1e-6,
    ),
    p = set_solver(
        model.p;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol        = 1e-2,
        atol        = 1e-6,
    ),
    energy = set_solver(
        model.energy;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol        = 1e-2,
        atol        = 1e-6,
    ),
)
runtime = set_runtime(
    iterations=100, time_step=1, write_interval=10)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, pressure)
initialise!(model.energy, h_inf)

Rx, Ry, Rz, Rp, Re = simple_rho_K_transonic!(model, thermodel, config)

plot(; xlims=(0,runtime.iterations))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Re), Re, yscale=:log10, label="energy")

# # # PROFILING CODE

# using Profile, PProf

# GC.gc()
# initialise!(model.U, velocity)
# initialise!(model.p, 0.0)

# Profile.Allocs.clear()
# Profile.Allocs.@profile sample_rate=1 begin 
#     Rx, Ry, Rp = simple!(model, config)
# end

# PProf.Allocs.pprof()
