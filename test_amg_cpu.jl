using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Inlet conditions
velocity = [0.5, 0.0, 0.0]
noSlip   = [0.0, 0.0, 0.0]
nu       = 1e-3
Re       = (0.2 * velocity[1]) / nu

model = Physics(
    time      = Transient(),
    fluid     = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy    = Energy{Isothermal}(),
    domain    = mesh_dev
)

BCs = assign(
    region = mesh_dev,
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

# ── Baseline: Cg + Jacobi (standard Krylov) ───────────────────────────────────

solvers_krylov = (
    U = SolverSetup(
        solver         = Bicgstab(),
        preconditioner = Jacobi(),
        convergence    = 1e-7,
        relax          = 1.0,
        rtol           = 0.0,
        atol           = 1e-5
    ),
    p = SolverSetup(
        solver         = Cg(),
        preconditioner = Jacobi(),
        convergence    = 1e-7,
        relax          = 1.0,
        rtol           = 0.0,
        atol           = 1e-5
    )
)

schemes = (
    U = Schemes(time=Euler, divergence=LUST, gradient=Gauss),
    p = Schemes(time=Euler, gradient=Gauss)
)

runtime = Runtime(iterations=50, write_interval=-1, time_step=0.0025)

config_krylov = Configuration(
    solvers=solvers_krylov, schemes=schemes, runtime=runtime,
    hardware=hardware, boundaries=BCs)

GC.gc(true)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

@info "=== Baseline: Cg + Jacobi (50 iterations) ==="
@time residuals_krylov = run!(model, config_krylov)

# ── AMG solver ────────────────────────────────────────────────────────────────

solvers_amg = (
    U = SolverSetup(
        solver         = Bicgstab(),
        preconditioner = Jacobi(),
        convergence    = 1e-7,
        relax          = 1.0,
        rtol           = 0.0,
        atol           = 1e-5
    ),
    p = SolverSetup(
        solver = AMG(
            smoother      = JacobiSmoother(2, 2/3, zeros(0)),
            cycle         = VCycle(),
            coarsening    = :SA,
            max_levels    = 20,
            coarsest_size = 50,
            pre_sweeps    = 2,
            post_sweeps   = 2,
            strength      = 0.25,
        ),
        preconditioner = Jacobi(),   # ignored by AMG; kept for API compat
        convergence    = 1e-7,
        relax          = 1.0,
        rtol           = 1e-3,
        atol           = 1e-5,
        itmax          = 10,
    )
)

config_amg = Configuration(
    solvers=solvers_amg, schemes=schemes, runtime=runtime,
    hardware=hardware, boundaries=BCs)

GC.gc(true)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

@info "=== AMG V-cycle (50 iterations, up to 10 cycles/solve) ==="
@time residuals_amg = run!(model, config_amg)

@info "Done. Compare elapsed times and allocations above."
