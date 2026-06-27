using XCALibre
using Test

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
mesh = UNV2D_mesh(mesh_file)

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

solvers = (
    T = SolverSetup(
        solver         = Cg(),
        preconditioner = Jacobi(),
        convergence    = 1e-8,
        relax          = 1.0,
    )
)
schemes = (T = Schemes(laplacian = Linear),)

# ── Case 1: Robin as Dirichlet (a=1, b=0) and as Zerogradient (a=0, b=1) ──────
# Robin with a=1, b=0, value=v  →  a·φ + b·∇φ·n = v  →  φ = v  (Dirichlet)
# Robin with a=0, b=1, value=0  →  ∇φ·n = 0            (Zerogradient)
BCs_robin = assign(
    region = mesh_dev,
    (
        T = [
            Robin(:left_wall,   a=1.0, b=0.0, value=50.0),
            Robin(:right_wall,  a=0.0, b=1.0, value=0.0),
            Robin(:bottom_wall, a=1.0, b=0.0, value=10.0),
            Robin(:upper_wall,  a=0.0, b=1.0, value=0.0),
        ],
    )
)
BCs_dirichlet = assign(
    region = mesh_dev,
    (
        T = [
            Dirichlet(:left_wall,   50.0),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, 10.0),
            Zerogradient(:upper_wall),
        ],
    )
)

cfg_r = Configuration(solvers=solvers, schemes=schemes,
    runtime=Runtime(iterations=1, write_interval=1, time_step=1),
    hardware=hardware, boundaries=BCs_robin)
cfg_d = Configuration(solvers=solvers, schemes=schemes,
    runtime=Runtime(iterations=1, write_interval=1, time_step=1),
    hardware=hardware, boundaries=BCs_dirichlet)

T_r = ScalarField(mesh_dev)
T_d = ScalarField(mesh_dev)
gamma = ConstantScalar(1.0)

eqn_r = (
    - Laplacian{schemes.T.laplacian}(gamma, T_r) == Source(ConstantScalar(0.0))
) → ScalarEquation(T_r, cfg_r.boundaries.T)

eqn_d = (
    - Laplacian{schemes.T.laplacian}(gamma, T_d) == Source(ConstantScalar(0.0))
) → ScalarEquation(T_d, cfg_d.boundaries.T)

discretise!(eqn_r, T_r, cfg_r)
apply_boundary_conditions!(eqn_r, cfg_r; time=0.0)
discretise!(eqn_d, T_d, cfg_d)
apply_boundary_conditions!(eqn_d, cfg_d; time=0.0)

@testset "Robin → Dirichlet/Zerogradient equivalence" begin
    @test eqn_r.equation.A.parent ≈ eqn_d.equation.A.parent
    @test eqn_r.equation.b       ≈ eqn_d.equation.b
end

# ── Case 2: mixed Robin (a=1, b=1) — analytic coefficient check ────────────────
# Unit mesh, 3×3 cells  →  cell width = 1/3, face–centre distance δ = 1/6
# denom = a·δ + b = 1/6 + 1 = 7/6
# Laplacian coefficient for interior faces: Γ·area/δ = 1·(1/3)/(1/3) = 1
# Robin contribution on left face of cell 1 (left-bottom):
#   ap_Robin = Γ·area·a / denom = 1·(1/3)·1 / (7/6) = 2/7
# Cell 1 diagonal = ap_interior_right + ap_interior_top + ap_Robin + ap_bottom(ZG=0)
#                 = 1 + 1 + 2/7 = 16/7
BCs_mixed = assign(
    region = mesh_dev,
    (
        T = [
            Robin(:left_wall,  a=1.0, b=1.0, value=100.0),
            Zerogradient(:right_wall),
            Zerogradient(:bottom_wall),
            Zerogradient(:upper_wall),
        ],
    )
)
cfg_m = Configuration(solvers=solvers, schemes=schemes,
    runtime=Runtime(iterations=1, write_interval=1, time_step=1),
    hardware=hardware, boundaries=BCs_mixed)

eqn_m = (
    - Laplacian{schemes.T.laplacian}(gamma, T_r) == Source(ConstantScalar(0.0))
) → ScalarEquation(T_r, cfg_m.boundaries.T)

discretise!(eqn_m, T_r, cfg_m)
apply_boundary_conditions!(eqn_m, cfg_m; time=0.0)

@testset "Robin mixed BC diagonal coefficient" begin
    area  = 1/3
    delta = 1/6
    ap_robin = (1.0 * area * 1.0) / (1.0 * delta + 1.0)
    expected_A11 = 1.0 + 1.0 + ap_robin   # right + top + Robin(left); bottom is ZG → 0
    @test eqn_m.equation.A.parent[1, 1] ≈ expected_A11
end
