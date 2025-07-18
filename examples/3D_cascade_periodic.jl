using XCALibre
# using Adapt
using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cascade_3D_periodic_2p5mm.unv"
# grid = "cascade_3D_periodic_4mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV3D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

periodic1 = construct_periodic(mesh, backend, :top, :bottom)
# periodic2 = construct_periodic(mesh, backend, :side1, :side2)
symmetric = Symmetry.([:side1, :side2])

velocity = [0.25, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs= assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:plate, [0.0, 0.0, 0.0]),
            periodic1...,
            symmetric...
            # periodic2...
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:plate),
            periodic1...,
            symmetric...
            # periodic2...
        ]
    )
)

schemes = (
    U = Schemes(divergence=Linear, gradient=Gauss),
    p = Schemes(gradient=Gauss)
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), #Cg(), # Bicgstab(), Gmres(), #Cg()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2
    )
)

runtime = Runtime(
    iterations=1000, time_step=1, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 353 iterations!

# using Plots
# fig = plot(; xlims=(0,runtime.iterations), ylims=(1e-10, 1e-4))
# plot!(fig, 1:runtime.iterations, residuals.Ux, yscale=:log10, label="Ux")
# plot!(fig, 1:runtime.iterations, residuals.Uy, yscale=:log10, label="Uy")
# plot!(fig, 1:runtime.iterations, residuals.p, yscale=:log10, label="p")
# fig