using Plots
using FVM_1D
using Krylov


# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

# Inlet conditions

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-2
Re = (0.2*velocity[1])/nu
temp = 300
Cp = 1005

model = dRANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Dirichlet(:bottom, velocity),
    Dirichlet(:top, velocity)
)

@assign! model h (
    Dirichlet(:inlet, Cp * temp),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, Cp * 305),
    Dirichlet(:top, Cp * temp),
    Dirichlet(:bottom, Cp * temp)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 100000.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.7,
    ),
    h = set_solver(
        model.h;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.3,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.3,
    )
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    h = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(divergence=Linear, gradient=Midpoint)
)

runtime = set_runtime(iterations=1000, write_interval=10, time_step=1.0)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 100000)
initialise!(model.h, 1005*temp)

Rx, Ry, Rh, Rp = dsimple!(model, config) #, pref=0.0)

plot(; xlims=(0,runtime.iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")