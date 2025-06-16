using Plots
using XCALibre
using CUDA

# quad and trig 40 and 100
mesh_file = "unv_sample_meshes/trig100.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = set_hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = 1*velocity[1]/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    Wall(:inlet, noSlip),
    Wall(:outlet, noSlip),
    Wall(:bottom, noSlip),
    Dirichlet(:top, velocity)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(gradient=Midpoint),
    p = set_schemes(gradient=Midpoint)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = set_runtime(iterations=1000, write_interval=1000, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, [0, 0, 0])
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config, pref=0.0) # 9.39k allocs

plot(1:runtime.iterations, Rx, yscale=:log10)
plot!(1:runtime.iterations, Ry, yscale=:log10)
plot!(1:runtime.iterations, Rp, yscale=:log10)