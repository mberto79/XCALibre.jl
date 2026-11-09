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
        atol=1e-6
    ),
    p = SolverSetup(
        solver=Cg(), # Richardson() for repeated V-cycles without Krylov acceleration
        # SmoothedAggregation(threshold=0.05): fewer iterations on anisotropic (boundary-layer)
        # pressure matrices; the default Pairwise() is cheaper to build and lighter on memory
        preconditioner=AMG(SmoothedAggregation(threshold=0.05);
            sweeps=10,
            coarsest=20000,
            # precision=MixedF32() in SolverSetup stores the coarse levels in Float32
            # NativeAMG(fuse=1) on a GPU (Pairwise() and Jacobi() only) stores no transfer matrices;
            # fuse=2 also applies the top coarse level through the finer matrix: less memory, more work
            native=NativeAMG(coarse=Smoothed(Jacobi(); sweeps=30))  # default: Direct()
        ),
        convergence=1e-7,
        relax=1.0,
        itmax=1000,
        # rtol=1e-4,
        rtol=0.0,
        atol=1e-6
    )
)

schemes = (
    U = Schemes(time=CrankNicolson, divergence=LUST, gradient=Gauss),
    p = Schemes(time=CrankNicolson, gradient=Gauss)
)

runtime = Runtime(iterations=1000, write_interval=50, time_step=0.0025)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)
