using XCALibre
using CUDA # uncomment to run on GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CUDABackend(); workgroup = 32
# backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Inlet conditions
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Wall(:cylinder, noSlip),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ],
        p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:cylinder),
                Zerogradient(:bottom),
                Zerogradient(:top)
                # Extrapolated(:bottom),
                # Extrapolated(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-6
    ),
    p = SolverSetup(
        solver      = AMG(
                        smoother      = JacobiSmoother(2, 2/3, zeros(0)),
                        # smoother      = Chebyshev(degree=2, lo=0.3, hi=1.1),
                        # smoother      = L1Jacobi(omega=1),
                        cycle         = VCycle(),
                        coarsening    = :RS, # :SA :RS
                        max_levels    = 15,
                        coarsest_size = 100,
                        pre_sweeps    = 2,
                        post_sweeps   = 1,
                        strength      = 0.0,
                        update_freq = 2,
                        krylov = :cg
                     ),
        # solver = Cg(),
        preconditioner = Jacobi(),   # ignored by AMG; kept for API compatibility
        convergence = 1e-7,
        relax       = 1.0,
        rtol        = 0.0,
        atol        = 1e-6,
        itmax       = 2000,
    )
)

timeScheme = Euler
timeScheme = CrankNicolson
schemes = (
    U = Schemes(time=timeScheme, divergence=LUST, gradient=Gauss),
    p = Schemes(time=timeScheme, gradient=Gauss)
)

iterations = 10000 # 10000 10
runtime = Runtime(
    iterations=iterations, write_interval=50, time_step=0.0025)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

@time residuals = run!(model, config)
