using Plots

using FVM_1D

using Krylov


# quad and trig 40 and 100
mesh_file = "unv_sample_meshes/trig100.unv"
mesh = build_mesh(mesh_file, scale=0.001)

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = 1*velocity[1]/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, noSlip),
    Dirichlet(:outlet, noSlip),
    Dirichlet(:bottom, noSlip),
    Dirichlet(:top, velocity)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.U;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = set_runtime(iterations=2000, write_interval=-1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, [0, 0, 0])
initialise!(model.p, 0.0)

Rx, Ry, Rp = isimple!(model, config; pref=0.0) # 9.39k allocs

plot(1:runtime.iterations, Rx, yscale=:log10)
plot!(1:runtime.iterations, Ry, yscale=:log10)
plot!(1:runtime.iterations, Rp, yscale=:log10)