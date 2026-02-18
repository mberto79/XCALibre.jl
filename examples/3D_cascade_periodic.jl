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
periodic2 = Symmetry.([:side1, :side2])

velocity = [0.25, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    # time = Transient(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(), # steady and unsteady tests
    # turbulence = LES{Smagorinsky}(),
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
            periodic2...
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:plate),
            periodic1...,
            periodic2...
        ],
        nut = [
            Dirichlet(:inlet, 0.0),
            Extrapolated(:outlet),
            Dirichlet(:plate, 0.0),
            periodic1...,
            periodic2...
        ]
    )
)

divergence = Linear # Upwind Linear LUST
schemes = (
    # # transient schemes
    # U = Schemes(time=Euler, divergence=divergence, gradient=Gauss),
    # p = Schemes(gradient=Gauss)

    # Steady schemes
    U = Schemes(divergence=divergence, gradient=Gauss),
    p = Schemes(gradient=Gauss)
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), #Cg(), # Bicgstab(), Gmres(), #Cg()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        # # transient setup
        # atol = 1e-6,
        # relax=1

        # steady setup
        relax       = 0.6,
        rtol = 1e-3
    ),
    p = SolverSetup(
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        # # transient setup
        # atol = 1e-6,
        # relax=1

        # steady setup
        relax       = 0.15,
        rtol = 1e-3
    )
)

runtime = Runtime(
    iterations=1000, time_step=1, write_interval=100) # steady setup
    # iterations=1000, time_step=1e-3, write_interval=100) # transient setup

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config, output=OpenFOAM()) # 353 iterations!

# using Plots
# fig = plot(; xlims=(0,runtime.iterations), ylims=(1e-10, 1e-4))
# plot!(fig, 1:runtime.iterations, residuals.Ux, yscale=:log10, label="Ux")
# plot!(fig, 1:runtime.iterations, residuals.Uy, yscale=:log10, label="Uy")
# plot!(fig, 1:runtime.iterations, residuals.p, yscale=:log10, label="p")
# fig