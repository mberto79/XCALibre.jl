using XCALibre
using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)
# backend = CPU(); workgroup = 1024; activate_multithread(backend)
backend = CUDABackend(); workgroup = 32
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3

model = Physics(
    time=Transient(),
    fluid=Fluid{Incompressible}(nu=nu),
    turbulence=RANS{Laminar}(),
    energy=Energy{Isothermal}(),
    domain=mesh_dev
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
            Extrapolated(:bottom),
            Extrapolated(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver=Bicgstab(),
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=1.0,
        rtol=0.0,
        atol=1e-5
    ),
    p = SolverSetup(
        solver=AMG(
            mode=:cg,
            # coarsening = RugeStuben(),
            coarsening=SmoothAggregation(
                strength_threshold=0.10,
                level_strength_thresholds=(0.10, 0.075, 0.05),
                max_prolongation_entries=2,
                aggressive_levels=1,
                aggressive_passes=1,
                coarse_drop_tolerances=(0.0, 0.01, 0.03, 0.05)
            ),
            smoother=AMGJacobi(omega=2/3),
            cycle=:V,
            presweeps=3,
            postsweeps=1,
            max_levels=10,
            min_coarse_rows=32,
            max_coarse_rows=512,
            adaptive_rebuild_factor=0.85,
        ),
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=1.0,
        itmax=200,
        rtol=0.0,
        atol=1e-5
    )
)

schemes = (
    U = Schemes(time=CrankNicolson, divergence=LUST, gradient=Gauss),
    p = Schemes(time=CrankNicolson, gradient=Gauss)
)

runtime = Runtime(iterations=2000, write_interval=50, time_step=0.0025)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)
