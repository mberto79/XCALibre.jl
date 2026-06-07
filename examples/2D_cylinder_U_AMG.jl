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
        solver=AMG(
            # mode = AMGSolver(),  
            mode = Cg(),  
            pre_sweeps=10,
            post_sweeps=10,
            #                     # GAMG-like iteration counts. Use SA/RS, not Geometric.
            # mode = Cg(),          # AMG-preconditioned CG (default; fastest single-device)
            # scale_correction = true  # default: GAMG energy-min coarse-correction scaling. Applies
            #                          # to AMGSolver only (gated; would break PCG's fixed-SPD prec).
            # coarse_solve = OnDevice(max_rows=512)  # default: coarsest solved on device (no host
            #                                        # round-trip) up to max_rows, else host LU
            # coarse_solve = CPU()                   # force coarsest solve on host (LU/QR)
            # coarse_solve = OnDeviceKrylov(), 
            coarse_solve = OnDeviceJacobi(iterations=30), 
            max_coarse_rows=20000,  

            #   truncate to a large, well-conditioned coarsest + solve it on-device with Cg/Bicgstab
            #   +Jacobi (no host-LU sync point). LARGE 3D wins: F1 1.68M -> 2.5x vs Cg+Jacobi baseline,
            #   1.5x vs same-mode OnDevice. Small/mildly-coarsened cases (this cylinder) LOSE — Jacobi-
            #   CG on the coarsest needs many iters. Set max_coarse_rows high enough to truncate at a
            #   sizeable coarsest. See src/Solve/AMG/AMG_OnDeviceKrylov_findings.md.
            # coarse_storage = Float32  # store the hierarchy in single precision (outer Krylov stays
            #   Float64, so iteration count + final residual are UNCHANGED). Halves V-cycle SpMV/
            #   smoother/RAP bandwidth. GPU-only win, scales with size: F1 1.68M pressure solve 1.77x
            #   (Cg) / 1.75x (AMGSolver) faster at identical iters. Valid in both modes. See
            #   src/Solve/AMG/AMG_mixed_precision_findings.md.
            # coarsening = Geometric(merge_levels=1)
            # fuse_levels = 1  # opt-in: matrix-free greenfield GPU V-cycle (GPU + Geometric +
            #   AMGJacobi only; default 0 = off). Erases the P/R transfer operators (formed on the
            #   fly) and refreshes coarse operators in-place on device each timestep, cutting the
            #   hierarchy footprint ~24% at iteration-parity (F1 1.68M: 518 vs 681 MB, Cg 156 vs 161
            #   iters). EXPERIMENTAL. WIN is VRAM on large 3D GPU transients near the memory ceiling;
            #   small/medium cases are ~10-15% SLOWER with no VRAM relief (refresh + matrix-free
            #   recompute overhead) -> leave at 0 unless VRAM-bound.
            # coarsening = RugeStuben()
            coarsening = SmoothAggregation(strength_threshold=0.05)  # opt-in: fewer iters on
            #                          # anisotropic (boundary-layer) pressure matrices; denser
            #                          # hierarchy (small wall-clock cost on warm-started solves)
            # coarsening = SmoothAggregation()
        ),
        # solver=Cg(),
        preconditioner=Jacobi(),
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
