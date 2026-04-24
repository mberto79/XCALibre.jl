using XCALibre
using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)
backend = CPU(); workgroup = 1024; activate_multithread(backend)
# backend = CUDABackend(); workgroup = 32
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
            coarsening=RugeStuben(
                strength_threshold=0.0, 
                strength_measure=:classical
                ),
            # coarsening=SmoothAggregation(
            #     strength_threshold=0.5,
            #     smoother_weight=0.67,
            #     truncate_factor=0.0,
            #     max_interp_entries=1,
            #     interpolation_passes=2,
            #     strength_measure=:classical,
            #     filter_weak_connections=true,
            #     near_nullspace=nothing
            #     ),
            smoother=AMGJacobi(),
            cycle=:V,
            presweeps=3,
            postsweeps=2,
            max_levels=10,
            min_coarse_rows=50,
            max_coarse_rows=256,
            adaptive_rebuild_factor=1,
            coarse_refresh_interval=1,
            numeric_refresh_rtol=1e-2,
            assume_fixed_pattern=true
        ),
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=1.0,
        itmax=500,
        rtol=0.0,
        atol=1e-5
    )
)

schemes = (
    U = Schemes(time=CrankNicolson, divergence=LUST, gradient=Gauss),
    p = Schemes(time=CrankNicolson, gradient=Gauss)
)

runtime = Runtime(iterations=500, write_interval=50, time_step=0.0025)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)
