# using Plots
using XCALibre
using CUDA

# quad and trig 40 and 100
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig100.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CUDABackend(); workgroup = 32
# backend = CPU(); workgroup = 1024 #; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
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

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Wall(:top, velocity),
            Wall(:outlet, noSlip),
            Wall(:bottom, noSlip),
            Wall(:inlet, noSlip)
        ],
        p = [
            # Extrapolated(:top),
            # Extrapolated(:inlet),
            # Extrapolated(:outlet),
            # Extrapolated(:bottom),
            Extrapolated(:top),
            Wall(:inlet),
            Wall(:outlet),
            Wall(:bottom)
        ]
    )
)

schemes = (
    U = Schemes(gradient=Midpoint),
    p = Schemes(gradient=Midpoint)
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # DILU(), # Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = Runtime(iterations=1000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, [0, 0, 0])
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config, pref=0.0) # 9.39k allocs

# plot(1:runtime.iterations, Rx, yscale=:log10)
# plot!(1:runtime.iterations, Ry, yscale=:log10)
# plot!(1:runtime.iterations, Rp, yscale=:log10)