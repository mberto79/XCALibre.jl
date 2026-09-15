using KernelAbstractions
using LinearAlgebra
using SparseMatricesCSR
using Test
using Adapt
using SparseArrays

function amg_test_matrix(T=Float64)
    i = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
    j = [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5]
    v = T[300, -100, -100, 200, -100, -100, 200, -100, -100, 200, -100, -100, 300]
    return SparseXCSR(sparsecsr(i, j, v, 5, 5)), T[200 * 500, 0, 0, 0, 200 * 100]
end

struct FakeCoarsening <: XCALibre.Solve.AbstractAMGCoarsening end
struct FakeSmoother <: XCALibre.Solve.AbstractAMGSmoother end

A, b = amg_test_matrix()
solver = AMG(mode=AMGSolver(), coarsening=SmoothAggregation(), smoother=AMGJacobi())
setup = SolverSetup(
    solver=solver,
    preconditioner=Jacobi(),
    convergence=1e-7,
    relax=1.0,
    rtol=1e-8,
    atol=1e-8
)

@test setup.solver isa AMG
@test setup.itmax == 200
@test setup.solver.pre_sweeps == 2
@test setup.solver.post_sweeps == 2
@test setup.solver.cycle isa VCycle

solver_jacobi_default = AMG()
@test solver_jacobi_default.mode isa Cg
@test solver_jacobi_default.scale_correction == true
@test AMG(scale_correction=false).scale_correction == false
@test solver_jacobi_default.pre_sweeps == 2
@test solver_jacobi_default.post_sweeps == 2
@test solver_jacobi_default.smoother.omega == 4/3
@test solver_jacobi_default.coarsening.strength_threshold == 0.0
@test solver_jacobi_default.coarsening.smoother_weight == 4/3
@test solver_jacobi_default.max_coarse_rows == 4096
@test solver_jacobi_default.coarse_refresh_interval == 1

solver_rs_default = AMG(coarsening=RugeStuben(), smoother=AMGJacobi())
@test solver_rs_default.coarsening.strength_threshold == 0.25
@test solver_rs_default.smoother.omega == 4/3

solver_type_api = AMG(
    mode=Cg(),
    coarsening=SmoothAggregation(strength_threshold=0.12),
    smoother=AMGChebyshev(),
    cycle=VCycle(),
    pre_sweeps=1,
    post_sweeps=2
)
@test solver_type_api.mode isa Cg
@test solver_type_api.cycle isa VCycle
@test solver_type_api.coarsening.strength_threshold == 0.12
@test solver_type_api.smoother isa AMGChebyshev
@test solver_type_api.smoother.degree == 3
@test solver_type_api.smoother.eig_ratio == 10.0
@test solver_type_api.pre_sweeps == 1
@test solver_type_api.post_sweeps == 2
@test AMGJacobi <: XCALibre.Solve.AbstractAMGGPUSmoother
@test AMGChebyshev <: XCALibre.Solve.AbstractAMGGPUSmoother
@test AMGGaussSeidel <: XCALibre.Solve.AbstractAMGCPUSmoother
@test AMGSOR <: XCALibre.Solve.AbstractAMGCPUSmoother
@test occursin("AMGJacobi", XCALibre.Solve._amg_gpu_smoother_names())
@test occursin("AMGChebyshev", XCALibre.Solve._amg_gpu_smoother_names())

solver_gs_default = AMG(mode=Cg(), smoother=AMGGaussSeidel())
@test solver_gs_default.smoother.sweep isa AMGSymmetricSweep
@test solver_gs_default.smoother.iterations == 1
@test solver_gs_default.pre_sweeps == 1
@test solver_gs_default.post_sweeps == 1

solver_gs_forward = AMG(smoother=AMGGaussSeidel(AMGForwardSweep(), iterations=2))
@test solver_gs_forward.smoother.sweep isa AMGForwardSweep
@test solver_gs_forward.smoother.iterations == 2

solver_sor_default = AMG(smoother=AMGSOR(1.1, AMGBackwardSweep()))
@test solver_sor_default.smoother.sweep isa AMGBackwardSweep
@test solver_sor_default.smoother.omega == 1.1
@test solver_sor_default.pre_sweeps == 1
@test solver_sor_default.post_sweeps == 1

@test_throws ArgumentError AMGGaussSeidel(iterations=0)
@test_throws ArgumentError AMGSOR(0.0)

controlled_sa = SmoothAggregation(strength_threshold=0.05)
@test controlled_sa.strength_threshold == 0.05

solver_cg_jacobi_default = AMG(mode=Cg(), smoother=AMGJacobi())
@test solver_cg_jacobi_default.pre_sweeps == 2
@test solver_cg_jacobi_default.post_sweeps == 2

solver_custom_sweeps = AMG(pre_sweeps=4, post_sweeps=4)
@test solver_custom_sweeps.pre_sweeps == 4
@test solver_custom_sweeps.post_sweeps == 4

solver_explicit_sweeps = AMG(pre_sweeps=2, post_sweeps=3)
@test solver_explicit_sweeps.pre_sweeps == 2
@test solver_explicit_sweeps.post_sweeps == 3

@test_throws ArgumentError AMG(cycle=Symbol("W"))
@test_throws ArgumentError AMG(mode=Symbol("cg"))
@test_throws ArgumentError AMG(coarsening=Symbol("smooth_aggregation"))
@test_throws ArgumentError AMG(smoother=Symbol("jacobi"))
@test_throws ArgumentError AMG(coarsening=FakeCoarsening())
@test_throws ArgumentError AMG(smoother=FakeSmoother())
@test_throws MethodError AMG(; (; Symbol("smoothing_" * "steps") => 1)...)
@test_throws MethodError AMG(; (; Symbol("pre" * "sweeps") => 1)...)
@test_throws MethodError AMG(; (; Symbol("connection_" * "strength") => 0.1)...)
@test_throws MethodError AMG(; (; Symbol("smoother_" * "omega") => 0.8)...)
@test_throws MethodError SmoothAggregation(max_prolongation_entries=1)
@test_throws MethodError SmoothAggregation(aggressive_levels=1)
@test_throws MethodError SmoothAggregation(interpolation=:unsmoothed)

config = Configuration(
    solvers=(T=setup,),
    schemes=(T=Schemes(),),
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    hardware=Hardware(backend=CPU(), workgroup=64),
    boundaries=(T=(),)
)

solver_controlled_sa = AMG(coarsening=controlled_sa, smoother=AMGJacobi(), max_coarse_rows=2)
ws_controlled_sa = _workspace(solver_controlled_sa, b)
ws_controlled_sa = XCALibre.Solve.update!(ws_controlled_sa, A, solver_controlled_sa, config)
@test ws_controlled_sa.hierarchy isa XCALibre.Solve.AMGHierarchy
@test ws_controlled_sa.hierarchy.operator_complexity >= 1

A_isolated = SparseXCSR(sparsecsr(collect(1:12), collect(1:12), ones(12), 12, 12))
b_isolated = ones(12)
isolated_coarsening = SmoothAggregation()
_, P_isolated, _ = XCALibre.Solve.build_prolongation(A_isolated, isolated_coarsening)
@test size(P_isolated, 2) == 12
solver_isolated_sa = AMG(coarsening=isolated_coarsening, smoother=AMGJacobi(), max_coarse_rows=4)
ws_isolated_sa = _workspace(solver_isolated_sa, b_isolated)
ws_isolated_sa = XCALibre.Solve.update!(ws_isolated_sa, A_isolated, solver_isolated_sa, config)
@test size(ws_isolated_sa.hierarchy.levels[end].A, 1) == 12
_, P_isolated_rs, _ = XCALibre.Solve.build_prolongation(A_isolated, RugeStuben())
@test size(P_isolated_rs, 2) == 12
solver_isolated_rs = AMG(coarsening=RugeStuben(), smoother=AMGJacobi(), max_coarse_rows=4)
ws_isolated_rs = _workspace(solver_isolated_rs, b_isolated)
ws_isolated_rs = XCALibre.Solve.update!(ws_isolated_rs, A_isolated, solver_isolated_rs, config)
@test size(ws_isolated_rs.hierarchy.levels[end].A, 1) == 12
@test XCALibre.Solve._is_diagonal_matrix(ws_isolated_sa.hierarchy.levels[end].A)

solver_rs = AMG(coarsening=RugeStuben(strength_threshold=0.2), smoother=AMGJacobi(), max_coarse_rows=2)
ws_rs = _workspace(solver_rs, b)
ws_rs = XCALibre.Solve.update!(ws_rs, A, solver_rs, config)
@test ws_rs.hierarchy isa XCALibre.Solve.AMGHierarchy
@test ws_rs.hierarchy.operator_complexity >= 1

ws = _workspace(setup.solver, b)
@test ws.solution !== ws.q
ws = XCALibre.Solve.update!(ws, A, setup.solver, config)
hierarchy_build = ws.hierarchy  # captured to verify later refreshes reuse (not rebuild) this object
@test ws.hierarchy isa XCALibre.Solve.AMGHierarchy
@test all(isconcretetype, fieldtypes(typeof(ws)))
@test all(isconcretetype, fieldtypes(typeof(ws.hierarchy)))
@test all(isconcretetype, fieldtypes(typeof(ws.hierarchy.levels[1])))
@test ws.refresh_count == 0  # initial build is not a refresh
@test length(ws.hierarchy.levels) >= 1
@test ws.hierarchy.operator_complexity >= 1
@test ws.hierarchy.grid_complexity >= 1

level_for_omega = ws.hierarchy.levels[1]
lambda_max_before = level_for_omega.lambda_max
# omega is lambda_max-scaled: ω_eff = omega/lambda_max (monotonic in omega)
level_for_omega.lambda_max = 4.0
@test XCALibre.Solve._level_jacobi_omega(AMGJacobi(omega=2 / 3), level_for_omega) ≈ (2 / 3) / 4
level_for_omega.lambda_max = 1.2
@test XCALibre.Solve._level_jacobi_omega(AMGJacobi(omega=2 / 3), level_for_omega) ≈ (2 / 3) / 1.2
# default omega=4/3 unchanged from the previous cap formula: 4/(3*lambda_max)
level_for_omega.lambda_max = 2.0
@test XCALibre.Solve._level_jacobi_omega(AMGJacobi(), level_for_omega) ≈ 4 / (3 * 2.0)
@test XCALibre.Solve._level_jacobi_omega(AMGJacobi(), level_for_omega) ≈ 2 / 3
# clamp keeps ω_eff < 2/lambda_max for SPD stability
level_for_omega.lambda_max = 1.0
@test XCALibre.Solve._level_jacobi_omega(AMGJacobi(omega=5.0), level_for_omega) < 2.0
level_for_omega.lambda_max = lambda_max_before

if length(ws.hierarchy.levels) > 1
    P = ws.hierarchy.levels[1].P
    @test P !== nothing
    @test nnz(P) >= size(P, 1)
    @test size(ws.hierarchy.levels[end].A, 1) < size(ws.hierarchy.levels[1].A, 1)
end

candidate = [1.0, 2.0, 3.0, 4.0, 5.0]
solver_sa_candidate = AMG(
    mode=AMGSolver(),
    coarsening=SmoothAggregation(near_nullspace=candidate),
    smoother=AMGJacobi()
)
ws_sa_candidate = _workspace(solver_sa_candidate, b)
ws_sa_candidate = XCALibre.Solve.update!(ws_sa_candidate, A, solver_sa_candidate, config)
if length(ws_sa_candidate.hierarchy.levels) > 1
    P_sa = Matrix(ws_sa_candidate.hierarchy.levels[1].P)
    @test maximum(abs, P_sa[:, 1]) != minimum(abs, P_sa[:, 1])
end

if length(ws_rs.hierarchy.levels) > 1
    P_rs = Matrix(ws_rs.hierarchy.levels[1].P)
    @test all(sum(abs, P_rs[i, :]) > 0 for i in axes(P_rs, 1))
    @test maximum(abs.(vec(sum(P_rs, dims=2)) .- 1)) < 1e-8
end

A2, _ = amg_test_matrix()
XCALibre.ModelFramework._nzval(A2) .= XCALibre.ModelFramework._nzval(A) .* 1.1
levels_before = length(ws.hierarchy.levels)
finest_before = copy(XCALibre.Solve._nzval(ws.hierarchy.levels[1].A))
coarse_before = length(ws.hierarchy.levels) > 1 ? copy(XCALibre.Solve._nzval(ws.hierarchy.levels[2].A)) : nothing
coarse_object_before = length(ws.hierarchy.levels) > 1 ? ws.hierarchy.levels[2].A : nothing
ws2 = XCALibre.Solve.update!(ws, A2, setup.solver, config)
@test length(ws2.hierarchy.levels) == levels_before
@test XCALibre.Solve._nzval(ws2.hierarchy.levels[1].A) != finest_before
@test ws2.refresh_count == 1  # same pattern, changed values: a refresh
@test ws2.hierarchy === hierarchy_build  # refreshed in place, not rebuilt
if length(ws2.hierarchy.levels) > 1
    # coarse_refresh_interval=1 (default): full refresh updates coarse operators in place
    @test ws2.hierarchy.levels[2].A === coarse_object_before
    @test XCALibre.Solve._nzval(ws2.hierarchy.levels[2].A) != coarse_before
end

solver_refresh = AMG(mode=AMGSolver(), coarsening=SmoothAggregation(), smoother=AMGJacobi(), coarse_refresh_interval=1)
ws_refresh = _workspace(solver_refresh, b)
ws_refresh = XCALibre.Solve.update!(ws_refresh, A, solver_refresh, config)
coarse_refresh_before = length(ws_refresh.hierarchy.levels) > 1 ? copy(XCALibre.Solve._nzval(ws_refresh.hierarchy.levels[2].A)) : nothing
coarse_acsc_before = ws_refresh.hierarchy.coarse_cpu.Acsc
coarse_acsc_colptr_before = ws_refresh.hierarchy.coarse_cpu.Acsc.colptr
coarse_acsc_rowval_before = ws_refresh.hierarchy.coarse_cpu.Acsc.rowval
hierarchy_refresh_before = ws_refresh.hierarchy
ws_refresh = XCALibre.Solve.update!(ws_refresh, A2, solver_refresh, config)
@test ws_refresh.refresh_count == 1
@test ws_refresh.hierarchy === hierarchy_refresh_before  # refreshed in place, not rebuilt
if length(ws_refresh.hierarchy.levels) > 1
    @test XCALibre.Solve._nzval(ws_refresh.hierarchy.levels[2].A) != coarse_refresh_before
    @test ws_refresh.hierarchy.coarse_cpu.Acsc === coarse_acsc_before
    @test ws_refresh.hierarchy.coarse_cpu.Acsc.colptr === coarse_acsc_colptr_before
    @test ws_refresh.hierarchy.coarse_cpu.Acsc.rowval === coarse_acsc_rowval_before
    @test Matrix(ws_refresh.hierarchy.coarse_cpu.Acsc) ≈ Matrix(ws_refresh.hierarchy.coarse_cpu.A)
end

# coarse_refresh_interval>1: refresh_count drives the finest-only vs full-coarse-refresh decision.
# Counter increments every update!; coarse operators only refresh on the interval-th call.
let solver_iv = AMG(mode=AMGSolver(), coarsening=SmoothAggregation(), smoother=AMGJacobi(), coarse_refresh_interval=2)
    ws_iv = _workspace(solver_iv, b)
    ws_iv = XCALibre.Solve.update!(ws_iv, A, solver_iv, config)
    @test ws_iv.refresh_count == 0
    if length(ws_iv.hierarchy.levels) > 1
        coarse_iv0 = copy(XCALibre.Solve._nzval(ws_iv.hierarchy.levels[2].A))
        A_iv = deepcopy(A); XCALibre.Solve._nzval(A_iv) .*= 1.05
        ws_iv = XCALibre.Solve.update!(ws_iv, A_iv, solver_iv, config)  # call 1: finest-only
        @test ws_iv.refresh_count == 1
        @test XCALibre.Solve._nzval(ws_iv.hierarchy.levels[2].A) == coarse_iv0  # coarse not yet refreshed
        ws_iv = XCALibre.Solve.update!(ws_iv, A_iv, solver_iv, config)  # call 2: full refresh
        @test ws_iv.refresh_count == 2
        @test XCALibre.Solve._nzval(ws_iv.hierarchy.levels[2].A) != coarse_iv0  # coarse refreshed on interval
    end
end

A_singular_coarse_csc = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1.0, 1.0, 2.0, 2.0], 2, 2)
A_singular_coarse = XCALibre.Solve._amg_matrix(A_singular_coarse_csc)
coarse_singular = XCALibre.Solve._empty_cpu_coarse_level(Float64)
XCALibre.Solve._refresh_coarse_cpu!(coarse_singular, A_singular_coarse)
@test coarse_singular.use_qr
coarse_singular_solution = XCALibre.Solve._coarse_solve!(coarse_singular, [2.0, 4.0])
@test A_singular_coarse_csc * coarse_singular_solution ≈ [2.0, 4.0]

A_pattern = SparseXCSR(sparsecsr(
    [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    [1, 3, 1, 2, 3, 1, 3, 4, 3, 4, 5, 4, 5],
    [300.0, -100.0, -100.0, 200.0, -100.0, -100.0, 200.0, -100.0, -100.0, 200.0, -100.0, -100.0, 300.0],
    5,
    5
))
hierarchy_before = ws2.hierarchy
ws3 = XCALibre.Solve.update!(ws2, A_pattern, setup.solver, config)
@test ws3.hierarchy !== hierarchy_before
@test ws3.hierarchy.rowptr_pattern == Int.(XCALibre.ModelFramework._rowptr(A_pattern))
@test ws3.hierarchy.colval_pattern == Int.(XCALibre.ModelFramework._colval(A_pattern))

x0 = zeros(eltype(b), length(b))
r0 = norm(b - Array(parent(A)) * x0)
fill!(ws.hierarchy.levels[1].x, 0)
XCALibre.Solve._apply_level_smoother!(ws.hierarchy, AMGJacobi(), ws.hierarchy.levels[1], b, 3)
rj = norm(b - Array(parent(A)) * ws.hierarchy.levels[1].x)
@test rj < r0

solver_chebyshev_smooth = AMG(mode=AMGSolver(), coarsening=SmoothAggregation(), smoother=AMGChebyshev(), max_coarse_rows=2)
ws_chebyshev_smooth = _workspace(solver_chebyshev_smooth, b)
ws_chebyshev_smooth = XCALibre.Solve.update!(ws_chebyshev_smooth, A, solver_chebyshev_smooth, config)
fill!(ws_chebyshev_smooth.hierarchy.levels[1].x, 0)
XCALibre.Solve._apply_level_smoother!(
    ws_chebyshev_smooth.hierarchy,
    solver_chebyshev_smooth.smoother,
    ws_chebyshev_smooth.hierarchy.levels[1],
    b,
    1
)
rc = norm(b - Array(parent(A)) * ws_chebyshev_smooth.hierarchy.levels[1].x)
@test rc < rj

for smoother in (AMGGaussSeidel(), AMGGaussSeidel(AMGForwardSweep()), AMGGaussSeidel(AMGBackwardSweep()), AMGSOR(1.0), AMGSOR(1.1, AMGForwardSweep()))
    solver_sweep = AMG(mode=AMGSolver(), coarsening=SmoothAggregation(), smoother=smoother, max_coarse_rows=2)
    ws_sweep = _workspace(solver_sweep, b)
    ws_sweep = XCALibre.Solve.update!(ws_sweep, A, solver_sweep, config)
    fill!(ws_sweep.hierarchy.levels[1].x, 0)
    XCALibre.Solve._apply_level_smoother!(ws_sweep.hierarchy, smoother, ws_sweep.hierarchy.levels[1], b, 1)
    rs = norm(b - Array(parent(A)) * ws_sweep.hierarchy.levels[1].x)
    @test rs < r0
end

x = zeros(eltype(b), length(b))
ws_solve = _workspace(setup.solver, b)
ws_solve = XCALibre.Solve.update!(ws_solve, A, setup.solver, config)
XCALibre.Solve.amg_solve!(ws_solve, ws_solve.hierarchy, setup.solver, ws_solve.hierarchy.levels[1].A, b, x; itmax=100, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x) / norm(b) < 1e-8
@test ws_solve.hierarchy.last_cycle_factor >= 0
@test ws_solve.iterations > 0
@test length(ws_solve.residual_history) == ws_solve.iterations + 1
@test ws_solve.residual_history[end] <= ws_solve.residual_history[1]

# scale_correction: AMGSolver converges to the same solution with sc on/off; Cg uses flexible PR+ β when sc is on
for sc in (false, true)
    solver_sc = AMG(mode=AMGSolver(), coarsening=SmoothAggregation(), smoother=AMGJacobi(), scale_correction=sc, max_coarse_rows=2)
    ws_sc = _workspace(solver_sc, b)
    ws_sc = XCALibre.Solve.update!(ws_sc, A, solver_sc, config)
    x_sc = zeros(eltype(b), length(b))
    XCALibre.Solve.amg_solve!(ws_sc, ws_sc.hierarchy, solver_sc, ws_sc.hierarchy.levels[1].A, b, x_sc; itmax=100, atol=1e-10, rtol=1e-10)
    @test norm(b - Array(parent(A)) * x_sc) / norm(b) < 1e-8
end

solver_cg = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGJacobi())
ws_cg = _workspace(solver_cg, b)
ws_cg = XCALibre.Solve.update!(ws_cg, A, solver_cg, config)
@test ws_cg.hierarchy.is_symmetric
x_cg = zeros(eltype(b), length(b))
XCALibre.Solve.amg_cg_solve!(ws_cg, ws_cg.hierarchy, solver_cg, ws_cg.hierarchy.levels[1].A, b, x_cg; itmax=50, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_cg) / norm(b) < 1e-8
@test ws_cg.iterations > 0
@test length(ws_cg.residual_history) == ws_cg.iterations + 1
@test ws_cg.residual_history[end] <= ws_cg.residual_history[1]

solver_cg_chebyshev = AMG(mode=Cg(), smoother=AMGChebyshev(), coarsening=SmoothAggregation())
ws_cg_chebyshev = _workspace(solver_cg_chebyshev, b)
ws_cg_chebyshev = XCALibre.Solve.update!(ws_cg_chebyshev, A, solver_cg_chebyshev, config)
x_cg_chebyshev = zeros(eltype(b), length(b))
XCALibre.Solve.amg_cg_solve!(
    ws_cg_chebyshev,
    ws_cg_chebyshev.hierarchy,
    solver_cg_chebyshev,
    ws_cg_chebyshev.hierarchy.levels[1].A,
    b,
    x_cg_chebyshev;
    itmax=50,
    atol=1e-8,
    rtol=1e-8
)
@test norm(b - Array(parent(A)) * x_cg_chebyshev) / norm(b) < 1e-8

x_rs = zeros(eltype(b), length(b))
ws_rs_solve = _workspace(solver_rs, b)
ws_rs_solve = XCALibre.Solve.update!(ws_rs_solve, A, solver_rs, config)
XCALibre.Solve.amg_solve!(ws_rs_solve, ws_rs_solve.hierarchy, solver_rs, ws_rs_solve.hierarchy.levels[1].A, b, x_rs; itmax=100, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_rs) / norm(b) < 1e-8
@test ws_rs_solve.iterations > 0

# Geometric (OpenFOAM-style) agglomeration
@test Geometric().merge_levels == 1
@test Geometric(merge_levels=2).merge_levels == 2
@test_throws ArgumentError Geometric(merge_levels=0)
solver_geo = AMG(mode=Cg(), coarsening=Geometric(), smoother=AMGJacobi())
@test solver_geo.coarsening isa Geometric
x_geo = zeros(eltype(b), length(b))
ws_geo = _workspace(solver_geo, b)
ws_geo = XCALibre.Solve.update!(ws_geo, A, solver_geo, config)
XCALibre.Solve.amg_cg_solve!(ws_geo, ws_geo.hierarchy, solver_geo, ws_geo.hierarchy.levels[1].A, b, x_geo; itmax=100, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_geo) / norm(b) < 1e-8

A_bad = SparseXCSR(sparsecsr([1, 1, 2], [1, 2, 2], [2.0, 1.0, 3.0], 2, 2))
b_bad = [1.0, 1.0]
ws_bad = _workspace(solver_cg, b_bad)
ws_bad = XCALibre.Solve.update!(ws_bad, A_bad, solver_cg, config)
@test !ws_bad.hierarchy.is_symmetric
@test_throws ArgumentError XCALibre.Solve.amg_cg_solve!(ws_bad, ws_bad.hierarchy, solver_cg, ws_bad.hierarchy.levels[1].A, b_bad, zeros(2); itmax=10, atol=1e-8, rtol=1e-8)

solver_nonsymmetric_solve = AMG(mode=AMGSolver(), coarsening=SmoothAggregation(), smoother=AMGJacobi(), max_coarse_rows=2)
ws_nonsymmetric_solve = _workspace(solver_nonsymmetric_solve, b_bad)
ws_nonsymmetric_solve = XCALibre.Solve.update!(ws_nonsymmetric_solve, A_bad, solver_nonsymmetric_solve, config)
x_bad = zeros(eltype(b_bad), length(b_bad))
XCALibre.Solve.amg_solve!(
    ws_nonsymmetric_solve,
    ws_nonsymmetric_solve.hierarchy,
    solver_nonsymmetric_solve,
    ws_nonsymmetric_solve.hierarchy.levels[1].A,
    b_bad,
    x_bad;
    itmax=20,
    atol=1e-8,
    rtol=1e-8
)
@test norm(b_bad - Array(parent(A_bad)) * x_bad) / norm(b_bad) < 1e-8

# NEW SECTION: mixed precision (coarse_storage=Float32) — API, type-stability, FP32 storage with
# FP64 outer correction reaches the FP64 tolerance and matches FP64 iteration count.
@test AMG(coarse_storage=Float32).coarse_storage === Float32
@test AMG().coarse_storage === Float64
@test_throws ArgumentError AMG(coarse_storage=Int)
for mode in (Cg(), AMGSolver())
    sref = AMG(mode=mode, coarsening=SmoothAggregation(), smoother=AMGJacobi())
    s32  = AMG(mode=mode, coarsening=SmoothAggregation(), smoother=AMGJacobi(), coarse_storage=Float32)
    runamg(s) = begin
        w = _workspace(s, b); w = XCALibre.Solve.update!(w, A, s, config)
        mixed = XCALibre.Solve._amg_mixed_precision(w.hierarchy)
        outer = mixed ? A : w.hierarchy.levels[1].A
        x = zeros(eltype(b), length(b))
        XCALibre.Solve._amg_solve_mode!(w, w.hierarchy, s, s.mode, outer, b, x; itmax=200, atol=1e-8, rtol=1e-8)
        (it=w.iterations, res=norm(b - Array(parent(A)) * x) / norm(b),
         store=eltype(XCALibre.Solve._nzval(w.hierarchy.levels[1].A)), mixed=mixed)
    end
    rref = runamg(sref); r32 = runamg(s32)
    @test rref.store === Float64 && !rref.mixed
    @test r32.store === Float32 && r32.mixed          # FP32 hierarchy actually built
    @test r32.res < 1e-8                              # FP64 outer correction reaches FP64 tol
    # Cg = fixed SPD preconditioner → exact iteration parity. AMGSolver = FP64 IR around an FP32
    # cycle → may differ by a few at tight tol (both converge); allow a small band.
    if mode isa Cg
        @test r32.it == rref.it
    else
        @test abs(r32.it - rref.it) <= max(2, rref.it ÷ 5)
    end
end

# FP32 REFRESH path (production: coarse_refresh_interval=1 re-runs update! every solve). Refresh with a
# changed matrix (same pattern) and confirm the FP32 hierarchy + FP64 outer still converge — exercises
# CPU _sync_storage_levels! and the FP64->FP32 coarse-operator sync that single-solve tests never hit.
let s = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGJacobi(), coarse_storage=Float32)
    w = _workspace(s, b); w = XCALibre.Solve.update!(w, A, s, config)
    h_before = w.hierarchy
    A2 = deepcopy(A); XCALibre.Solve._nzval(A2) .*= 1.07   # same pattern → refresh (not rebuild)
    w = XCALibre.Solve.update!(w, A2, s, config)
    @test w.refresh_count >= 1            # took a refresh path
    @test w.hierarchy === h_before        # refreshed in place, not rebuilt
    x = zeros(eltype(b), length(b))
    XCALibre.Solve._amg_solve_mode!(w, w.hierarchy, s, s.mode, A2, b, x; itmax=200, atol=1e-8, rtol=1e-8)
    @test norm(b - Array(parent(A2)) * x) / norm(b) < 1e-7   # FP32-refreshed operators stay correct
end

# NEW SECTION: Float32-valued mesh regression (single precision Int32+Float32, mixed Int64+Float32).
# SuiteSparse (CHOLMOD/UMFPACK/SPQR) factorizes only Float64, so a Float32 coarsest used to crash the
# host coarse direct solve at workspace creation regardless of coarse_storage. The coarse direct solve
# now runs in Float64; the rest of the hierarchy stays at the (Float32) working precision. Also asserts
# coarse_storage is clamped to <= working precision (no silent Float64 upcast on a Float32 mesh).
function amg_test_matrix_i32(::Type{T}) where {T}
    i = Int32[1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
    j = Int32[1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5]
    v = T[300, -100, -100, 200, -100, -100, 200, -100, -100, 200, -100, -100, 300]
    return SparseXCSR(sparsecsr(i, j, v, 5, 5)), T[200 * 500, 0, 0, 0, 200 * 100]
end
for makeA in (amg_test_matrix, amg_test_matrix_i32)
    Af, bf = makeA(Float32)
    @test eltype(XCALibre.Solve._nzval(Af)) === Float32
    for storage in (Float32, Float64), mode in (Cg(), AMGSolver())
        s = AMG(mode=mode, coarsening=SmoothAggregation(), smoother=AMGJacobi(), coarse_storage=storage)
        w = _workspace(s, bf)                                   # used to crash here (placeholder lu)
        w = XCALibre.Solve.update!(w, Af, s, config)            # full hierarchy build
        @test eltype(XCALibre.Solve._nzval(w.hierarchy.levels[1].A)) === Float32   # storage clamped
        @test !XCALibre.Solve._amg_mixed_precision(w.hierarchy)                    # no upcast
        @test eltype(w.hierarchy.coarse_cpu.Acsc) === Float64                      # Float64 direct solve
        x = zeros(Float32, length(bf))
        XCALibre.Solve._amg_solve_mode!(w, w.hierarchy, s, s.mode, w.hierarchy.levels[1].A, bf, x; itmax=200, atol=1f-6, rtol=1f-6)
        @test norm(bf - Array(parent(Af)) * x) / norm(bf) < 1e-4   # Float32 working-precision floor
    end
end

try
    using CUDA
    if CUDA.functional()
        backend_gpu = CUDABackend()
        config_gpu = Configuration(
            solvers=(T=setup,),
            schemes=(T=Schemes(),),
            runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
            hardware=Hardware(backend=backend_gpu, workgroup=64),
            boundaries=(T=(),)
        )

        ig, jg, vg = findnz(parent(A))
        A_gpu = XCALibre.ModelFramework._build_A(backend_gpu, ig, jg, vg, size(A, 1))
        b_gpu = adapt(backend_gpu, b)
        x_gpu = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
        ws_gpu = _workspace(setup.solver, b_gpu)
        ws_gpu = XCALibre.Solve.update!(ws_gpu, A_gpu, setup.solver, config_gpu)
        @test XCALibre.Solve._nzval(ws_gpu.hierarchy.levels[1].A) isa CuArray
        @test ws_gpu.hierarchy.coarse_cpu.rhs isa Vector

        i2, j2, v2 = findnz(parent(A2))
        A2_gpu = XCALibre.ModelFramework._build_A(backend_gpu, i2, j2, v2, size(A2, 1))
        ws_gpu = XCALibre.Solve.update!(ws_gpu, A2_gpu, setup.solver, config_gpu)
        @test Array(XCALibre.Solve._nzval(ws_gpu.hierarchy.levels[1].A)) == Array(XCALibre.ModelFramework._nzval(A2))

        XCALibre.Solve.amg_solve!(ws_gpu, ws_gpu.hierarchy, setup.solver, ws_gpu.hierarchy.levels[1].A, b_gpu, x_gpu; itmax=20, atol=1e-8, rtol=1e-8)
        @test norm(Array(b_gpu) - Array(parent(A2)) * Array(x_gpu)) / norm(Array(b_gpu)) < 1e-6

        solver_cg_gpu = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGJacobi())
        ws_cg_gpu = _workspace(solver_cg_gpu, b_gpu)
        ws_cg_gpu = XCALibre.Solve.update!(ws_cg_gpu, A_gpu, solver_cg_gpu, config_gpu)
        x_cg_gpu = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
        XCALibre.Solve.amg_cg_solve!(ws_cg_gpu, ws_cg_gpu.hierarchy, solver_cg_gpu, ws_cg_gpu.hierarchy.levels[1].A, b_gpu, x_cg_gpu; itmax=20, atol=1e-8, rtol=1e-8)
        @test norm(Array(b_gpu) - Array(parent(A)) * Array(x_cg_gpu)) / norm(Array(b_gpu)) < 1e-6

        # mixed precision on device: FP32 cuSPARSE hierarchy + FP64 outer (raw A) reaches FP64 tol
        for mode_gpu in (Cg(), AMGSolver())
            s32g = AMG(mode=mode_gpu, coarsening=SmoothAggregation(), smoother=AMGJacobi(), coarse_storage=Float32)
            w32g = _workspace(s32g, b_gpu)
            w32g = XCALibre.Solve.update!(w32g, A_gpu, s32g, config_gpu)
            @test XCALibre.Solve._amg_mixed_precision(w32g.hierarchy)
            @test XCALibre.Solve._nzval(w32g.hierarchy.levels[1].A) isa CuArray{Float32}
            x32g = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
            XCALibre.Solve._amg_solve_mode!(w32g, w32g.hierarchy, s32g, s32g.mode, A_gpu, b_gpu, x32g; itmax=50, atol=1e-8, rtol=1e-8)
            @test norm(Array(b_gpu) - Array(parent(A)) * Array(x32g)) / norm(Array(b_gpu)) < 1e-6
            # FP32 device refresh (production path): re-run update! with the changed A2_gpu, then solve.
            # Exercises _device_copyto! FP64->FP32 broadcast + device FP32 RAP + mixed coarse rebuild.
            w32g = XCALibre.Solve.update!(w32g, A2_gpu, s32g, config_gpu)
            x32g2 = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
            XCALibre.Solve._amg_solve_mode!(w32g, w32g.hierarchy, s32g, s32g.mode, A2_gpu, b_gpu, x32g2; itmax=50, atol=1e-8, rtol=1e-8)
            @test norm(Array(b_gpu) - Array(parent(A2)) * Array(x32g2)) / norm(Array(b_gpu)) < 1e-6
        end

        # greenfield matrix-free split-precision device REFRESH (device/5). The coarse RAP scatter does a
        # GPU atomic add across the precision boundary (F64 finest -> F32 coarse); guard the type match and
        # that a refreshed mixed-prec state converges as well as the F64 build (coarse F32 = zero iter cost).
        # Needs a matrix large enough to coarsen at max_coarse=64, so build a 2D Poisson here (A is 5x5).
        let nx = 40
            np = nx*nx; ip = Int[]; jp = Int[]; vp = Float64[]
            pid(i,j) = (j-1)*nx + i
            for j in 1:nx, i in 1:nx
                k = pid(i,j); d = 0.0
                for (di,dj) in ((1,0),(-1,0),(0,1),(0,-1))
                    (1 <= i+di <= nx && 1 <= j+dj <= nx) || continue
                    push!(ip,k); push!(jp,pid(i+di,j+dj)); push!(vp,-1.0); d += 1.0
                end
                push!(ip,k); push!(jp,k); push!(vp,d)
            end
            g1 = SparseXCSR(sparsecsr(ip, jp, vp, np, np))
            g2 = deepcopy(g1)
            let rp = XCALibre.Solve._rowptr(g2), cv = XCALibre.Solve._colval(g2), nz = XCALibre.Solve._nzval(g2)
                for r in 1:XCALibre.Solve._m(g2), p in rp[r]:(rp[r+1]-1)
                    nz[p] *= (cv[p] == r ? 1.25 : 0.85)
                end
            end
            rm = XCALibre.Solve.validate_refresh(g1, g2, 1, backend_gpu; max_coarse=64, coarse_storage=Float32)
            @test rm.relerr < 1e-5
            cg64 = XCALibre.Solve.validate_refresh_convergence(g1, g2, 1, backend_gpu; max_coarse=64, coarse_storage=Float64)
            cg32 = XCALibre.Solve.validate_refresh_convergence(g1, g2, 1, backend_gpu; max_coarse=64, coarse_storage=Float32)
            @test cg32.converged && cg32.iters <= cg64.iters + 1
        end

        # T3: GPU fuse_levels equivalence. fl=0 (materialised) vs fl=1 (matrix-free) on the same F64
        # device system must agree in iters (+-1) and solution (rel < 1e-8); fl=2 (fused coarse) converges.
        # Screened Poisson (diag = neighbors + 1): strictly diagonally dominant so the omega=4/3 Jacobi
        # smoother is stable (the pure Poisson diag==sum|offdiag| makes the Geometric AMG cycle diverge).
        let nx = 48
            np = nx*nx; ip = Int[]; jp = Int[]; vp = Float64[]
            pid(i, j) = (j-1)*nx + i
            for j in 1:nx, i in 1:nx
                k = pid(i, j); d = 0.0
                for (di, dj) in ((1,0),(-1,0),(0,1),(0,-1))
                    (1 <= i+di <= nx && 1 <= j+dj <= nx) || continue
                    push!(ip, k); push!(jp, pid(i+di, j+dj)); push!(vp, -1.0); d += 1.0
                end
                push!(ip, k); push!(jp, k); push!(vp, d + 1.0)
            end
            A_t3 = XCALibre.ModelFramework._build_A(backend_gpu, ip, jp, vp, np)
            b_t3 = adapt(backend_gpu, ones(np))
            Ah_t3 = sparse(ip, jp, vp, np, np)
            run_fl(fl) = begin
                s = AMG(mode=Cg(), coarsening=Geometric(merge_levels=1), smoother=AMGJacobi(),
                        max_coarse_rows=128, fuse_levels=fl)
                ws = _workspace(s, b_t3)
                ws = XCALibre.Solve.update!(ws, A_t3, s, config_gpu)
                x = KernelAbstractions.zeros(backend_gpu, Float64, np)
                XCALibre.Solve._amg_solve_mode!(ws, ws.hierarchy, s, s.mode, A_t3, b_t3, x; itmax=200, atol=0.0, rtol=1e-8)
                (iters=ws.iterations, x=Array(x), mf=ws.hierarchy isa XCALibre.Solve.MatrixFreeHierarchy)
            end
            r0 = run_fl(0); r1 = run_fl(1)
            @test !r0.mf                                   # fl=0 -> materialised
            @test r1.mf                                    # fl=1 -> matrix-free
            @test abs(r1.iters - r0.iters) <= 1
            @test norm(r1.x - r0.x) / norm(r0.x) < 1e-8
            r2 = run_fl(2)
            @test r2.mf
            @test norm(ones(np) - Ah_t3 * r2.x) / sqrt(np) < 1e-6
        end

        solver_chebyshev_gpu = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGChebyshev())
        ws_chebyshev_gpu = _workspace(solver_chebyshev_gpu, b_gpu)
        ws_chebyshev_gpu = XCALibre.Solve.update!(ws_chebyshev_gpu, A_gpu, solver_chebyshev_gpu, config_gpu)
        x_chebyshev_gpu = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
        XCALibre.Solve.amg_cg_solve!(
            ws_chebyshev_gpu,
            ws_chebyshev_gpu.hierarchy,
            solver_chebyshev_gpu,
            ws_chebyshev_gpu.hierarchy.levels[1].A,
            b_gpu,
            x_chebyshev_gpu;
            itmax=20,
            atol=1e-8,
            rtol=1e-8
        )
        @test norm(Array(b_gpu) - Array(parent(A)) * Array(x_chebyshev_gpu)) / norm(Array(b_gpu)) < 1e-6

        solver_gs_gpu = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGGaussSeidel())
        err = try
            XCALibre.Solve.setup_hierarchy(A_gpu, solver_gs_gpu, backend_gpu; log_diagnostics=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin("sequential CPU AMG smoother", message)
        @test occursin("AMGJacobi", message)
        @test occursin("AMGChebyshev", message)
    end
catch err
    @info "Skipping CUDA AMG test" exception=(err, catch_backtrace())
end
