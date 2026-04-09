
# # 2D Cylinder — AMG with PCG Acceleration
#
# This example demonstrates using the Algebraic Multigrid (AMG) solver with
# Preconditioned Conjugate Gradient (PCG) acceleration for the pressure equation
# in a transient incompressible flow around a 2D cylinder.
#
# ## AMG as a standalone solver vs as a Krylov preconditioner
#
# With `krylov = :cg` (default), each pressure solve runs Preconditioned CG where
# **one AMG V-cycle is the preconditioner** M⁻¹:
#
#   x_{k+1} = x_k + α_k p_k        (Krylov-optimal step)
#   p_{k+1} = M⁻¹ r_{k+1} + β_k p_k  (conjugate search direction)
#
# With `krylov = :none`, each call runs V-cycles as plain Richardson iteration:
#
#   x_{k+1} = x_k + ω D⁻¹ (b - A x_k)  (fixed-point / Richardson step)
#
# ## Performance trade-off
#
# PCG uses the full Krylov subspace to pick an optimal step, so it reaches a
# lower residual than Richardson with the same number of V-cycles — typically
# 10–100× better inner convergence per call. However, each PCG step costs one
# extra SpMV (for A*p) and two global reductions (dot products), adding ~25%
# overhead per timestep on CPU. On GPU the dot-product syncs are proportionally
# more expensive; the overhead is typically 30–50%.
#
# **When to use `krylov = :cg`:**
# - When a tight inner pressure solve is needed (low `atol`, high `itmax`): PCG
#   reaches lower residuals per V-cycle and pays back its overhead.
# - When the outer PISO/SIMPLE loop uses only a few pressure correctors: better
#   inner convergence reduces the number of outer iterations required.
# - For anisotropic meshes or irregular refinement where AMG quality is limited
#   and κ(M⁻¹A) is large: PCG's O(√κ) vs O(κ) advantage is most pronounced.
#
# **When to use `krylov = :none`:**
# - When the inner solve is inexpensive (high `rtol` or small `itmax`): Richardson
#   exits in 1–5 V-cycles and PCG overhead is not recovered.
# - On GPU with many short solves: each dot-product sync costs PCIe round-trip
#   latency; Richardson with coarse tolerance is often faster.
# - For rapid prototyping or debugging where simplest-possible behaviour is wanted.

using XCALibre
using CUDA # comment out if running on CPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# ── Backend selection ──────────────────────────────────────────────────────────
backend = CUDABackend(); workgroup = 32
# backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# ── Flow conditions ────────────────────────────────────────────────────────────
velocity = [0.5, 0.0, 0.0]
noSlip   = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.01 * velocity[1]) / nu   # D = 10 mm

model = Physics(
    time       = Transient(),
    fluid      = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy     = Energy{Isothermal}(),
    domain     = mesh_dev
)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet,    velocity),
            Zerogradient(:outlet),
            Wall(:cylinder,      noSlip),
            Extrapolated(:bottom),
            Extrapolated(:top)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet,   0.0),
            Wall(:cylinder),
            Extrapolated(:bottom),
            Extrapolated(:top)
        ]
    )
)

# ── Solvers ────────────────────────────────────────────────────────────────────
#
# Two pressure solver configurations are shown; uncomment the one you want.
#
# Option A (krylov = :cg): PCG with AMG preconditioner.
#   Each step: 1 V-cycle (preconditioner apply) + 1 SpMV (A*p) + 2 dot products.
#   Achieves Krylov-optimal convergence — reaches lower residuals per V-cycle than
#   Richardson. Adds ~25-50% runtime overhead due to the extra SpMV and dot syncs.
#   Best when tight inner tolerance is needed (small atol, large itmax).
#
# Option B (krylov = :none): Richardson (plain V-cycle) iteration.
#   Each step: 1 V-cycle. Cheaper per step; competitive when the solve exits early
#   (large rtol or warm start). Use when throughput matters more than accuracy.
#
# The `itmax` field controls the maximum number of steps (V-cycles for both paths).
# The `update_freq` field controls how often the coarse-level hierarchy is rebuilt;
# update_freq=2 halves the Galerkin rebuild cost with negligible accuracy impact.

solvers = (
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
            # ── Smoother ──────────────────────────────────────────────────────
            # Damped Jacobi: loops=sweeps per level, omega=damping (2/3 optimal
            # for FVM Laplacians). Alternatively use Chebyshev(degree=2).
            smoother      = JacobiSmoother(3, 2/3, zeros(0)),

            # ── Cycle type ─────────────────────────────────────────────────
            cycle         = VCycle(),   # or WCycle() for harder problems

            # ── Coarsening ─────────────────────────────────────────────────
            # :SA (Smoothed Aggregation) is the default and works well for
            # near-isotropic FVM Laplacians. :RS (Ruge-Stüben) is an
            # alternative for strongly anisotropic problems.
            coarsening    = :RS,

            # ── Hierarchy parameters ────────────────────────────────────────
            max_levels    = 15,
            coarsest_size = 100,   # direct LU solve below this size
            pre_sweeps    = 2,
            post_sweeps   = 1,
            strength      = 0.0,   # keep all connections (correct for FVM Laplacian)
            update_freq   = 2,     # refresh coarse hierarchy every 2 outer iterations

            # ── Krylov acceleration ─────────────────────────────────────────
            # :cg   → PCG (default). Krylov-optimal; best inner convergence per
            #         V-cycle. Adds ~25–50% overhead (extra SpMV + 2 dot syncs).
            #         Preferred when tight atol is needed or itmax is large.
            # :none → Richardson (plain V-cycle loop). Lower overhead per step;
            #         fastest when early exit via rtol dominates.
            krylov        = :cg,
        ),
        preconditioner = Jacobi(),   # required by API; AMG ignores this field
        convergence    = 1e-7,
        relax          = 1.0,
        rtol           = 0.0,
        atol           = 1e-5,
        itmax          = 20,         # max PCG steps per outer pressure solve
    )
)

# ── Numerical schemes ──────────────────────────────────────────────────────────
schemes = (
    U = Schemes(time=CrankNicolson, divergence=LUST, gradient=Gauss),
    p = Schemes(time=CrankNicolson, gradient=Gauss)
)

# ── Runtime ────────────────────────────────────────────────────────────────────
iterations = 10000
runtime = Runtime(iterations=iterations, write_interval=50, time_step=0.0025)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

@time residuals = run!(model, config)
