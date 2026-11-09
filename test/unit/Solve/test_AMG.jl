using XCALibre
using KernelAbstractions
using LinearAlgebra
using SparseMatricesCSR
using Test
using Adapt
using SparseArrays

const SV = XCALibre.Solve

function amg_test_matrix(T=Float64, TI=Int)
    i = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
    j = [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5]
    v = T[300, -100, -100, 200, -100, -100, 200, -100, -100, 200, -100, -100, 300]
    return SparseXCSR(sparsecsr(TI.(i), TI.(j), v, TI(5), TI(5))), T[200 * 500, 0, 0, 0, 200 * 100]
end

struct FakeSmoother <: SV.AbstractAMGSmoother end
struct FakeGPU <: KernelAbstractions.GPU end

A, b = amg_test_matrix()
setup = SolverSetup(
    solver=Richardson(),
    preconditioner=AMG(SmoothedAggregation(); refresh=1),
    convergence=1e-7,
    relax=1.0,
    rtol=1e-8,
    atol=1e-8
)

@test setup.solver isa Richardson
@test setup.preconditioner isa AMG
@test setup.itmax == 1000
@test setup.preconditioner.sweeps == Sweeps(2, 2)

# NEW SECTION: constructors and defaults
amg_default = AMG()
@test amg_default.coarsening isa Pairwise && amg_default.coarsening.merges == 2
@test amg_default.smoother isa Jacobi && amg_default.smoother.weight == 4/3
@test amg_default.sweeps.pre == amg_default.sweeps.post == 2
@test amg_default.coarsest == 4096
@test amg_default.refresh == 5
@test amg_default.native.coarse == Direct(max_rows=512)
@test amg_default.native.scale_correction && amg_default.native.fuse == 0
@test SV._amg_storage(amg_default) === Float64
@test AMG(native=NativeAMG(scale_correction=false)).native.scale_correction == false

sa_default = AMG(SmoothedAggregation())
@test sa_default.coarsening.threshold == 0.0
@test sa_default.coarsening.weight == 4/3
@test AMG(RugeStuben()).coarsening.threshold == 0.25

amg_cheb = AMG(SmoothedAggregation(threshold=0.12); smoother=Chebyshev(), sweeps=(1, 2))
@test amg_cheb.coarsening.threshold == 0.12
@test amg_cheb.smoother.degree == 3 && amg_cheb.smoother.eig_ratio == 10.0
@test amg_cheb.sweeps.pre == 1 && amg_cheb.sweeps.post == 2
@test AMG(smoother=Chebyshev()).sweeps == Sweeps(1, 1)
@test AMG(smoother=GaussSeidel()).sweeps == Sweeps(1, 1)
@test AMG(smoother=SSOR(1.1)).smoother.omega == 1.1
@test AMG(sweeps=4).sweeps == Sweeps(4, 4)
@test AMG(sweeps=Sweeps(2, 3)).sweeps.post == 3
@test SV._pre_sweeps(Sweeps(1, 1; grow=1), 3) == 3
@test SV._pre_sweeps(Sweeps(1, 1; grow=1), 6) == 4
@test Pairwise(2).merges == 2

@test GaussSeidel() isa SV.SweepSmoother && SSOR(1.1) isa SV.SweepSmoother
@test !(Jacobi() isa SV.SweepSmoother) && !(Chebyshev() isa SV.SweepSmoother)
@test SV._check_smoother_backend(CPU(), GaussSeidel()) isa GaussSeidel
@test SV._check_smoother_backend(FakeGPU(), Jacobi()) isa Jacobi
gs_gpu_err = try SV._check_smoother_backend(FakeGPU(), GaussSeidel()); nothing catch e; e end
@test gs_gpu_err isa ArgumentError && occursin("Jacobi()", sprint(showerror, gs_gpu_err))

@test_throws ArgumentError SOR(0.0)
@test_throws ArgumentError Pairwise(0)
@test_throws ArgumentError Chebyshev(degree=0)
@test_throws ArgumentError Sweeps(0, 0)
@test_throws ArgumentError Smoothed(GaussSeidel())
@test_throws ArgumentError Iterative(Richardson())
@test_throws ArgumentError AMG(coarsest=0)
@test_throws ArgumentError AMG(refresh=0)
@test_throws ArgumentError AMG(Symbol("smooth_aggregation"))
@test_throws ArgumentError AMG(smoother=Symbol("jacobi"))
@test_throws ArgumentError AMG(smoother=FakeSmoother())
@test_throws ArgumentError AMG(native=Direct())
@test_throws ArgumentError SV._workspace(AMG(native=NativeAMG(coarse=Iterative())), b; solver=Cg())
@test_throws MethodError SmoothedAggregation(max_prolongation_entries=1)
@test_throws MethodError SmoothedAggregation(aggressive_levels=1)
@test_throws MethodError SmoothedAggregation(interpolation=:unsmoothed)

@test SV._amg_storage(SolverSetup(solver=Cg(), preconditioner=AMG(), convergence=1e-7, relax=1.0,
    precision=MixedF32()).preconditioner) === Float32

# deprecated spellings still map to the current settings
@testset "AMG deprecated spellings" begin
    Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        @test Geometric(merge_levels=2) == Pairwise(2)
        @test SmoothAggregation(strength_threshold=0.05).threshold == 0.05
        @test AMGJacobi(omega=1.0).weight == 1.0
        @test AMGGaussSeidel() isa SymGaussSeidel
        @test AMGSOR(1.1, AMGForwardSweep()) == SOR(1.1)
        @test_throws ArgumentError AMGGaussSeidel(iterations=2)
        @test_throws ArgumentError AMGSOR(1.0, AMGBackwardSweep())
        @test AMG(coarsening=SmoothAggregation(), pre_sweeps=1).sweeps.pre == 1
        @test AMG(max_coarse_rows=7, coarse_refresh_interval=3).coarsest == 7
        @test AMG(coarse_solve=OnHost()).native.coarse == Direct(max_rows=0)
        @test AMG(mode=Bicgstab()) isa SV._AMGMode
        @test_throws ArgumentError AMG(coarse_storage=Float32)
        @test_throws ArgumentError AMG(smoothing_steps=1)
    end
end

config = Configuration(
    solvers=(T=setup,),
    schemes=(T=Schemes(),),
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    hardware=Hardware(backend=CPU(), workgroup=64),
    boundaries=(T=(),)
)

# NEW SECTION: hierarchy setup and refresh
controlled_sa = SmoothedAggregation(threshold=0.05)
solver_controlled_sa = AMG(controlled_sa; coarsest=2)
ws_controlled_sa = _workspace(solver_controlled_sa, b)
ws_controlled_sa = SV.update!(ws_controlled_sa, A, solver_controlled_sa, config)
@test ws_controlled_sa.hierarchy isa SV.AMGHierarchy
@test ws_controlled_sa.hierarchy.operator_complexity >= 1

A_isolated = SparseXCSR(sparsecsr(collect(1:12), collect(1:12), ones(12), 12, 12))
b_isolated = ones(12)
_, P_isolated, _ = SV.build_prolongation(A_isolated, SmoothedAggregation())
@test size(P_isolated, 2) == 12
solver_isolated_sa = AMG(SmoothedAggregation(); coarsest=4)
ws_isolated_sa = _workspace(solver_isolated_sa, b_isolated)
ws_isolated_sa = SV.update!(ws_isolated_sa, A_isolated, solver_isolated_sa, config)
@test size(ws_isolated_sa.hierarchy.levels[end].A, 1) == 12
_, P_isolated_rs, _ = SV.build_prolongation(A_isolated, RugeStuben())
@test size(P_isolated_rs, 2) == 12
solver_isolated_rs = AMG(RugeStuben(); coarsest=4)
ws_isolated_rs = _workspace(solver_isolated_rs, b_isolated)
ws_isolated_rs = SV.update!(ws_isolated_rs, A_isolated, solver_isolated_rs, config)
@test size(ws_isolated_rs.hierarchy.levels[end].A, 1) == 12
@test SV._is_diagonal_matrix(ws_isolated_sa.hierarchy.levels[end].A)

solver_rs = AMG(RugeStuben(threshold=0.2); coarsest=2)
ws_rs = _workspace(solver_rs, b)
ws_rs = SV.update!(ws_rs, A, solver_rs, config)
@test ws_rs.hierarchy isa SV.AMGHierarchy
@test ws_rs.hierarchy.operator_complexity >= 1

ws = _workspace(setup.preconditioner, b; solver=Richardson())
@test ws.solution !== ws.q
ws = SV.update!(ws, A, setup.preconditioner, config)
hierarchy_build = ws.hierarchy  # captured to verify later refreshes reuse (not rebuild) this object
@test ws.hierarchy isa SV.AMGHierarchy
@test all(isconcretetype, fieldtypes(typeof(ws)))
@test all(isconcretetype, fieldtypes(typeof(ws.hierarchy)))
@test all(isconcretetype, fieldtypes(typeof(ws.hierarchy.levels[1])))
@test ws.refresh_count == 0  # initial build is not a refresh

# an Int32 equation matrix keeps every stored hierarchy index at Int32
A32, _ = amg_test_matrix(Float64, Int32)
ws32 = SV.update!(_workspace(setup.preconditioner, b, SV._index_type(A32)), A32, setup.preconditioner, config)
lv32 = ws32.hierarchy.host_levels[1]
@test SV._index_type(A32) == Int32
@test eltype(lv32.A.colval) == eltype(lv32.P.rowptr) == eltype(lv32.diagonal_index) == eltype(lv32.aggregate_ids) == Int32
@test eltype(ws32.hierarchy.rowptr_pattern) == Int32
@test length(ws.hierarchy.levels) >= 1
@test ws.hierarchy.operator_complexity >= 1
@test ws.hierarchy.grid_complexity >= 1

level_for_omega = ws.hierarchy.levels[1]
lambda_max_before = level_for_omega.lambda_max
# weight is lambda_max-scaled: ω_eff = weight/lambda_max (monotonic in weight)
level_for_omega.lambda_max = 4.0
@test SV._level_jacobi_omega(Jacobi(weight=2 / 3), level_for_omega) ≈ (2 / 3) / 4
level_for_omega.lambda_max = 1.2
@test SV._level_jacobi_omega(Jacobi(weight=2 / 3), level_for_omega) ≈ (2 / 3) / 1.2
level_for_omega.lambda_max = 2.0
@test SV._level_jacobi_omega(Jacobi(), level_for_omega) ≈ 2 / 3
# clamp keeps ω_eff < 2/lambda_max for SPD stability
level_for_omega.lambda_max = 1.0
@test SV._level_jacobi_omega(Jacobi(weight=5.0), level_for_omega) < 2.0
level_for_omega.lambda_max = lambda_max_before

if length(ws.hierarchy.levels) > 1
    P = ws.hierarchy.levels[1].P
    @test P !== nothing
    @test nnz(P) >= size(P, 1)
    @test size(ws.hierarchy.levels[end].A, 1) < size(ws.hierarchy.levels[1].A, 1)
end

candidate = [1.0, 2.0, 3.0, 4.0, 5.0]
solver_sa_candidate = AMG(SmoothedAggregation(near_nullspace=candidate))
ws_sa_candidate = _workspace(solver_sa_candidate, b; solver=Richardson())
ws_sa_candidate = SV.update!(ws_sa_candidate, A, solver_sa_candidate, config)
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
finest_before = copy(SV._nzval(ws.hierarchy.levels[1].A))
coarse_before = length(ws.hierarchy.levels) > 1 ? copy(SV._nzval(ws.hierarchy.levels[2].A)) : nothing
coarse_object_before = length(ws.hierarchy.levels) > 1 ? ws.hierarchy.levels[2].A : nothing
ws2 = SV.update!(ws, A2, setup.preconditioner, config)
@test length(ws2.hierarchy.levels) == levels_before
@test SV._nzval(ws2.hierarchy.levels[1].A) != finest_before
@test ws2.refresh_count == 1  # same pattern, changed values: a refresh
@test ws2.hierarchy === hierarchy_build  # refreshed in place, not rebuilt
if length(ws2.hierarchy.levels) > 1
    # refresh=1: full refresh updates coarse operators in place
    @test ws2.hierarchy.levels[2].A === coarse_object_before
    @test SV._nzval(ws2.hierarchy.levels[2].A) != coarse_before
end

solver_refresh = AMG(SmoothedAggregation(); refresh=1)
ws_refresh = _workspace(solver_refresh, b; solver=Richardson())
ws_refresh = SV.update!(ws_refresh, A, solver_refresh, config)
coarse_refresh_before = length(ws_refresh.hierarchy.levels) > 1 ? copy(SV._nzval(ws_refresh.hierarchy.levels[2].A)) : nothing
coarse_acsc_before = ws_refresh.hierarchy.coarse_cpu.Acsc
coarse_acsc_colptr_before = ws_refresh.hierarchy.coarse_cpu.Acsc.colptr
coarse_acsc_rowval_before = ws_refresh.hierarchy.coarse_cpu.Acsc.rowval
hierarchy_refresh_before = ws_refresh.hierarchy
ws_refresh = SV.update!(ws_refresh, A2, solver_refresh, config)
@test ws_refresh.refresh_count == 1
@test ws_refresh.hierarchy === hierarchy_refresh_before  # refreshed in place, not rebuilt
if length(ws_refresh.hierarchy.levels) > 1
    @test SV._nzval(ws_refresh.hierarchy.levels[2].A) != coarse_refresh_before
    @test ws_refresh.hierarchy.coarse_cpu.Acsc === coarse_acsc_before
    @test ws_refresh.hierarchy.coarse_cpu.Acsc.colptr === coarse_acsc_colptr_before
    @test ws_refresh.hierarchy.coarse_cpu.Acsc.rowval === coarse_acsc_rowval_before
    @test Matrix(ws_refresh.hierarchy.coarse_cpu.Acsc) ≈ Matrix(ws_refresh.hierarchy.coarse_cpu.A)
end

# refresh>1: every update! counts; coarse operators only refresh on the refresh-th call
let solver_iv = AMG(SmoothedAggregation(); refresh=2)
    ws_iv = _workspace(solver_iv, b; solver=Richardson())
    ws_iv = SV.update!(ws_iv, A, solver_iv, config)
    @test ws_iv.refresh_count == 0
    if length(ws_iv.hierarchy.levels) > 1
        coarse_iv0 = copy(SV._nzval(ws_iv.hierarchy.levels[2].A))
        A_iv = deepcopy(A); SV._nzval(A_iv) .*= 1.05
        ws_iv = SV.update!(ws_iv, A_iv, solver_iv, config)  # call 1: finest-only
        @test ws_iv.refresh_count == 1
        @test SV._nzval(ws_iv.hierarchy.levels[2].A) == coarse_iv0
        ws_iv = SV.update!(ws_iv, A_iv, solver_iv, config)  # call 2: full refresh
        @test ws_iv.refresh_count == 2
        @test SV._nzval(ws_iv.hierarchy.levels[2].A) != coarse_iv0
    end
end

A_singular_coarse_csc = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1.0, 1.0, 2.0, 2.0], 2, 2)
A_singular_coarse = SV._amg_matrix(A_singular_coarse_csc)
coarse_singular = SV._empty_cpu_coarse_level(Float64)
SV._refresh_coarse_cpu!(coarse_singular, A_singular_coarse)
@test coarse_singular.use_qr
coarse_singular_solution = SV._coarse_solve!(coarse_singular, [2.0, 4.0])
@test A_singular_coarse_csc * coarse_singular_solution ≈ [2.0, 4.0]

A_pattern = SparseXCSR(sparsecsr(
    [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    [1, 3, 1, 2, 3, 1, 3, 4, 3, 4, 5, 4, 5],
    [300.0, -100.0, -100.0, 200.0, -100.0, -100.0, 200.0, -100.0, -100.0, 200.0, -100.0, -100.0, 300.0],
    5,
    5
))
hierarchy_before = ws2.hierarchy
ws3 = SV.update!(ws2, A_pattern, setup.preconditioner, config)
@test ws3.hierarchy !== hierarchy_before
@test ws3.hierarchy.rowptr_pattern == Int.(XCALibre.ModelFramework._rowptr(A_pattern))
@test ws3.hierarchy.colval_pattern == Int.(XCALibre.ModelFramework._colval(A_pattern))

# NEW SECTION: smoothers
x0 = zeros(eltype(b), length(b))
r0 = norm(b - Array(parent(A)) * x0)
fill!(ws.hierarchy.levels[1].x, 0)
SV._apply_level_smoother!(ws.hierarchy, Jacobi(), ws.hierarchy.levels[1], b, 3)
rj = norm(b - Array(parent(A)) * ws.hierarchy.levels[1].x)
@test rj < r0

solver_chebyshev_smooth = AMG(SmoothedAggregation(); smoother=Chebyshev(), coarsest=2)
ws_chebyshev_smooth = _workspace(solver_chebyshev_smooth, b; solver=Richardson())
ws_chebyshev_smooth = SV.update!(ws_chebyshev_smooth, A, solver_chebyshev_smooth, config)
fill!(ws_chebyshev_smooth.hierarchy.levels[1].x, 0)
SV._apply_level_smoother!(ws_chebyshev_smooth.hierarchy, solver_chebyshev_smooth.smoother,
    ws_chebyshev_smooth.hierarchy.levels[1], b, 1)
rc = norm(b - Array(parent(A)) * ws_chebyshev_smooth.hierarchy.levels[1].x)
@test rc < rj

for smoother in (GaussSeidel(), SymGaussSeidel(), SOR(1.1), SSOR(1.1))
    solver_sweep = AMG(SmoothedAggregation(); smoother, coarsest=2)
    ws_sweep = _workspace(solver_sweep, b; solver=Richardson())
    ws_sweep = SV.update!(ws_sweep, A, solver_sweep, config)
    fill!(ws_sweep.hierarchy.levels[1].x, 0)
    SV._apply_level_smoother!(ws_sweep.hierarchy, smoother, ws_sweep.hierarchy.levels[1], b, 1)
    rs = norm(b - Array(parent(A)) * ws_sweep.hierarchy.levels[1].x)
    @test rs < r0
end

# NEW SECTION: solves
x = zeros(eltype(b), length(b))
ws_solve = _workspace(setup.preconditioner, b; solver=Richardson())
ws_solve = SV.update!(ws_solve, A, setup.preconditioner, config)
SV.amg_solve!(ws_solve, ws_solve.hierarchy, setup.preconditioner, ws_solve.hierarchy.levels[1].A, b, x; itmax=100, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x) / norm(b) < 1e-8
@test ws_solve.hierarchy.last_cycle_factor >= 0
@test ws_solve.iterations > 0
@test length(ws_solve.residual_history) == ws_solve.iterations + 1
@test ws_solve.residual_history[end] <= ws_solve.residual_history[1]

# scale_correction: Richardson converges to the same solution with sc on/off; Cg uses flexible PR+ β when sc is on
for sc in (false, true)
    solver_sc = AMG(SmoothedAggregation(); coarsest=2, native=NativeAMG(scale_correction=sc))
    ws_sc = _workspace(solver_sc, b; solver=Richardson())
    ws_sc = SV.update!(ws_sc, A, solver_sc, config)
    x_sc = zeros(eltype(b), length(b))
    SV.amg_solve!(ws_sc, ws_sc.hierarchy, solver_sc, ws_sc.hierarchy.levels[1].A, b, x_sc; itmax=100, atol=1e-10, rtol=1e-10)
    @test norm(b - Array(parent(A)) * x_sc) / norm(b) < 1e-8
end

solver_cg = AMG(SmoothedAggregation())
ws_cg = _workspace(solver_cg, b)
ws_cg = SV.update!(ws_cg, A, solver_cg, config)
@test ws_cg.hierarchy.is_symmetric
x_cg = zeros(eltype(b), length(b))
SV.amg_cg_solve!(ws_cg, ws_cg.hierarchy, solver_cg, ws_cg.hierarchy.levels[1].A, b, x_cg; itmax=50, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_cg) / norm(b) < 1e-8
@test ws_cg.iterations > 0
@test length(ws_cg.residual_history) == ws_cg.iterations + 1
@test ws_cg.residual_history[end] <= ws_cg.residual_history[1]

solver_cg_chebyshev = AMG(SmoothedAggregation(); smoother=Chebyshev())
ws_cg_chebyshev = _workspace(solver_cg_chebyshev, b)
ws_cg_chebyshev = SV.update!(ws_cg_chebyshev, A, solver_cg_chebyshev, config)
x_cg_chebyshev = zeros(eltype(b), length(b))
SV.amg_cg_solve!(ws_cg_chebyshev, ws_cg_chebyshev.hierarchy, solver_cg_chebyshev,
    ws_cg_chebyshev.hierarchy.levels[1].A, b, x_cg_chebyshev; itmax=50, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_cg_chebyshev) / norm(b) < 1e-8

x_rs = zeros(eltype(b), length(b))
ws_rs_solve = _workspace(solver_rs, b; solver=Richardson())
ws_rs_solve = SV.update!(ws_rs_solve, A, solver_rs, config)
SV.amg_solve!(ws_rs_solve, ws_rs_solve.hierarchy, solver_rs, ws_rs_solve.hierarchy.levels[1].A, b, x_rs; itmax=100, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_rs) / norm(b) < 1e-8
@test ws_rs_solve.iterations > 0

# Pairwise agglomeration: aggregate-list hierarchy with a permuted finest level,
# so the outer operator is the given matrix, not levels[1].A; it always coarsens at least once
solver_geo = AMG(Pairwise())
x_geo = zeros(eltype(b), length(b))
ws_geo = _workspace(solver_geo, b)
ws_geo = SV.update!(ws_geo, A, solver_geo, config)
@test ws_geo.hierarchy isa SV.MatrixFreeHierarchy
@test ws_geo.hierarchy.coarse_n < size(A, 1)
@test SV.outer_operator(ws_geo.hierarchy, A) === A
SV.amg_cg_solve!(ws_geo, ws_geo.hierarchy, solver_geo, SV.outer_operator(ws_geo.hierarchy, A), b, x_geo; itmax=100, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_geo) / norm(b) < 1e-8

A_bad = SparseXCSR(sparsecsr([1, 1, 2], [1, 2, 2], [2.0, 1.0, 3.0], 2, 2))
b_bad = [1.0, 1.0]
ws_bad = _workspace(solver_cg, b_bad)
ws_bad = SV.update!(ws_bad, A_bad, solver_cg, config)
@test !ws_bad.hierarchy.is_symmetric
@test_throws ArgumentError SV.amg_cg_solve!(ws_bad, ws_bad.hierarchy, solver_cg, ws_bad.hierarchy.levels[1].A, b_bad, zeros(2); itmax=10, atol=1e-8, rtol=1e-8)

solver_nonsymmetric_solve = AMG(SmoothedAggregation(); coarsest=2)
ws_nonsymmetric_solve = _workspace(solver_nonsymmetric_solve, b_bad; solver=Richardson())
ws_nonsymmetric_solve = SV.update!(ws_nonsymmetric_solve, A_bad, solver_nonsymmetric_solve, config)
x_bad = zeros(eltype(b_bad), length(b_bad))
SV.amg_solve!(ws_nonsymmetric_solve, ws_nonsymmetric_solve.hierarchy, solver_nonsymmetric_solve,
    ws_nonsymmetric_solve.hierarchy.levels[1].A, b_bad, x_bad; itmax=20, atol=1e-8, rtol=1e-8)
@test norm(b_bad - Array(parent(A_bad)) * x_bad) / norm(b_bad) < 1e-8

# NEW SECTION: BiCGStab outer iteration, valid for a non-symmetric operator
# `A_bad` is reused on purpose: `Cg()` refuses it above, and BiCGStab must solve it.
@test SV._amg_solver_name(Bicgstab()) == "bicgstab"

# workspace aliasing: BiCGStab's three extra vectors may alias nothing, `svec` above all not
# `solution`, which `solve_system!` solves in place
solver_bicg = AMG(SmoothedAggregation(); coarsest=2)
ws_bicg = _workspace(solver_bicg, b_bad; solver=Bicgstab())
ws_bicg = SV.update!(ws_bicg, A_bad, solver_bicg, config)
@test ws_bicg.svec !== ws_bicg.solution
@test ws_bicg.shadow !== ws_bicg.residual
@test ws_bicg.t !== ws_bicg.q
@test length(ws_bicg.svec) == length(b_bad)
@test length(ws_bicg.shadow) == length(b_bad)
@test length(ws_bicg.t) == length(b_bad)

# ...and allocated only for BiCGStab; zero-length keeps the field type concrete
ws_cg_mem = _workspace(AMG(SmoothedAggregation()), b_bad; solver=Cg())
@test length(ws_cg_mem.shadow) == 0
@test length(ws_cg_mem.t) == 0
@test length(ws_cg_mem.svec) == 0
@test typeof(ws_cg_mem.shadow) === typeof(ws_cg_mem.residual)

# it solves the matrix `Cg()` refuses; symmetry is measured whatever the outer iteration
@test !ws_bicg.hierarchy.is_symmetric
x_bicg = zeros(eltype(b_bad), length(b_bad))
SV.amg_bicgstab_solve!(ws_bicg, ws_bicg.hierarchy, solver_bicg, ws_bicg.hierarchy.levels[1].A,
    b_bad, x_bicg; itmax=50, atol=1e-10, rtol=1e-10)
@test norm(b_bad - Array(parent(A_bad)) * x_bicg) / norm(b_bad) < 1e-8
@test ws_bicg.converged
@test x_bicg ≈ Array(parent(A_bad)) \ b_bad rtol=1e-6

# the solution vector survives an in-place solve through `workspace.solution`
ws_alias = _workspace(solver_bicg, b_bad; solver=Bicgstab())
ws_alias = SV.update!(ws_alias, A_bad, solver_bicg, config)
copyto!(ws_alias.solution, zeros(eltype(b_bad), length(b_bad)))
SV.amg_bicgstab_solve!(ws_alias, ws_alias.hierarchy, solver_bicg, ws_alias.hierarchy.levels[1].A,
    b_bad, ws_alias.solution; itmax=50, atol=1e-10, rtol=1e-10)
@test norm(b_bad - Array(parent(A_bad)) * Array(ws_alias.solution)) / norm(b_bad) < 1e-8

# early exit when the initial guess is already converged
ws_exact = _workspace(solver_bicg, b_bad; solver=Bicgstab())
ws_exact = SV.update!(ws_exact, A_bad, solver_bicg, config)
x_exact = Array(parent(A_bad)) \ b_bad
SV.amg_bicgstab_solve!(ws_exact, ws_exact.hierarchy, solver_bicg, ws_exact.hierarchy.levels[1].A,
    b_bad, copy(x_exact); itmax=50, atol=1e-8, rtol=1e-8)
@test ws_exact.iterations == 0
@test ws_exact.converged

# it also solves a symmetric system
solver_bicg_sym = AMG(SmoothedAggregation())
ws_bicg_sym = _workspace(solver_bicg_sym, b; solver=Bicgstab())
ws_bicg_sym = SV.update!(ws_bicg_sym, A, solver_bicg_sym, config)
x_bicg_sym = zeros(eltype(b), length(b))
SV.amg_bicgstab_solve!(ws_bicg_sym, ws_bicg_sym.hierarchy, solver_bicg_sym,
    ws_bicg_sym.hierarchy.levels[1].A, b, x_bicg_sym; itmax=100, atol=1e-10, rtol=1e-10)
@test norm(b - Array(parent(A)) * x_bicg_sym) / norm(b) < 1e-8

# NEW SECTION: sequential smoothers
# every sweep converges; sweeps skip `j == i`, so a mishandled diagonal (converging to the wrong answer) is caught
for smoother in (GaussSeidel(), SymGaussSeidel(), SOR(1.1), SSOR(1.1))
    solver_gs = AMG(SmoothedAggregation(); smoother)
    ws_gs = _workspace(solver_gs, b; solver=Bicgstab())
    ws_gs = SV.update!(ws_gs, A, solver_gs, config)
    x_gs = zeros(eltype(b), length(b))
    SV.amg_bicgstab_solve!(ws_gs, ws_gs.hierarchy, solver_gs, ws_gs.hierarchy.levels[1].A,
        b, x_gs; itmax=100, atol=1e-10, rtol=1e-10)
    @test norm(b - Array(parent(A)) * x_gs) / norm(b) < 1e-8
end

# a zero diagonal row is skipped, not divided by
A_zerodiag = SparseXCSR(sparsecsr([1, 1, 2, 2], [1, 2, 1, 2], [0.0, 1.0, 1.0, 2.0], 2, 2))
b_zerodiag = [1.0, 1.0]
solver_zd = AMG(SmoothedAggregation(); smoother=GaussSeidel(), coarsest=2)
ws_zd = _workspace(solver_zd, b_zerodiag; solver=Bicgstab())
ws_zd = SV.update!(ws_zd, A_zerodiag, solver_zd, config)
x_zd = zeros(eltype(b_zerodiag), length(b_zerodiag))
SV.amg_bicgstab_solve!(ws_zd, ws_zd.hierarchy, solver_zd, ws_zd.hierarchy.levels[1].A,
    b_zerodiag, x_zd; itmax=20, atol=1e-8, rtol=1e-8)
@test all(isfinite, Array(x_zd))

# a row with no stored diagonal gets a substituted one, so it is not skipped; the gap is on row 2
# because only a later row can misread the CSR arrays
A_md_dense = [ 4.0 -1.0  0.0  0.0;
              -1.0  0.0 -1.0  0.0;
               0.0 -1.0  4.0 -1.0;
               0.0  0.0 -1.0  4.0]
A_md = SparseXCSR(sparsecsr([1, 1, 2, 2, 3, 3, 3, 4, 4],
                            [1, 2, 1, 3, 2, 3, 4, 3, 4],
                            [4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0], 4, 4))
b_md = [1.0, 2.0, 3.0, 4.0]
function reference_gauss_seidel!(x, Ad, d, b, order)
    for i in order
        sigma = sum(Ad[i, j] * x[j] for j in eachindex(x) if j != i)
        iszero(d[i]) || (x[i] = (b[i] - sigma) / d[i])
    end
    return x
end
for (smoother_md, orders) in ((GaussSeidel(), (1:4,)), (SymGaussSeidel(), (1:4, 4:-1:1)))
    solver_md = AMG(SmoothedAggregation(); smoother=smoother_md, coarsest=10)
    ws_md = _workspace(solver_md, b_md; solver=Richardson())
    ws_md = SV.update!(ws_md, A_md, solver_md, config)
    lvl_md = ws_md.hierarchy.levels[1]
    @test SV._diag_index(lvl_md.A)[2] == 0    # premise: no stored (2,2)
    fill!(lvl_md.x, 0.0)
    SV._apply_level_smoother_impl!(ws_md.hierarchy, smoother_md, lvl_md, b_md, 1)
    x_ref_md = zeros(4)
    for order in orders
        reference_gauss_seidel!(x_ref_md, A_md_dense, Array(lvl_md.diagonal), b_md, order)
    end
    @test Array(lvl_md.x) ≈ x_ref_md rtol=1e-12
end

# NEW SECTION: BiCGStab with the scale-corrected V-cycle converges on the TRUE residual
# 1D upwind convection-diffusion: non-symmetric and several levels deep, so scale_correction acts
n_cd = 60
I_cd = Int[]; J_cd = Int[]; V_cd = Float64[]
for i in 1:n_cd
    push!(I_cd, i); push!(J_cd, i); push!(V_cd, 3.0)
    i > 1    && (push!(I_cd, i); push!(J_cd, i - 1); push!(V_cd, -2.0))   # upwind side
    i < n_cd && (push!(I_cd, i); push!(J_cd, i + 1); push!(V_cd, -1.0))
end
A_cd = SparseXCSR(sparsecsr(I_cd, J_cd, V_cd, n_cd, n_cd))
A_cd_dense = Array(parent(A_cd))
@test A_cd_dense != transpose(A_cd_dense)
b_cd = collect(range(1.0, 2.0, length=n_cd))
x_cd_by_sc = Dict{Bool, Vector{Float64}}()
for sc in (false, true)
    solver_cd = AMG(SmoothedAggregation(); coarsest=4, native=NativeAMG(scale_correction=sc))
    ws_cd = _workspace(solver_cd, b_cd; solver=Bicgstab())
    ws_cd = SV.update!(ws_cd, A_cd, solver_cd, config)
    @test length(ws_cd.hierarchy.levels) > 1
    x_cd = zeros(n_cd)
    SV.amg_bicgstab_solve!(ws_cd, ws_cd.hierarchy, solver_cd, ws_cd.hierarchy.levels[1].A,
        b_cd, x_cd; itmax=200, atol=1e-10, rtol=1e-10)
    true_rel_cd = norm(b_cd - A_cd_dense * x_cd) / norm(b_cd)
    @test ws_cd.converged
    @test true_rel_cd < 1e-8
    # the reported residual is the TRUE residual, not the recurrence's estimate
    @test abs(ws_cd.last_relative_residual - true_rel_cd) <= 1e-3 * true_rel_cd + 1e-14
    x_cd_by_sc[sc] = x_cd
end
@test x_cd_by_sc[true] ≈ x_cd_by_sc[false] rtol=1e-6

# a warm start under a large ||b|| still iterates: a floor scaled by ||b|| used to declare breakdown.
# The asserts pin both halves: r0^2 < eps*||b|| (the old floor tripped) and r0 > atol.
solver_ws = AMG(SmoothedAggregation())
b_ws = b_cd .* 1e5
ws_ws = _workspace(solver_ws, b_ws; solver=Bicgstab())
ws_ws = SV.update!(ws_ws, A_cd, solver_ws, config)
x_ws = A_cd_dense \ b_ws
x_ws .+= 1e-5 .* sin.(range(0, 3pi, length=n_cd))
r0_ws = norm(b_ws - A_cd_dense * x_ws)
@test r0_ws > 2e-6
@test r0_ws^2 < eps(Float64) * norm(b_ws)
SV.amg_bicgstab_solve!(ws_ws, ws_ws.hierarchy, solver_ws, ws_ws.hierarchy.levels[1].A,
    b_ws, x_ws; itmax=200, atol=2e-6, rtol=0.0)
@test ws_ws.iterations >= 1
@test norm(b_ws - A_cd_dense * x_ws) <= 2e-6
@test ws_ws.converged

# NEW SECTION: coarse solves (Smoothed and Iterative run as Direct on a CPU backend)
for (coarse, outer) in ((Direct(max_rows=0), Cg()), (Smoothed(), Cg()), (Iterative(), Richardson())),
        coarsening in (SmoothedAggregation(), Pairwise())
    s = AMG(coarsening; coarsest=2, native=NativeAMG(; coarse))
    w = SV.update!(_workspace(s, b; solver=outer), A, s, config)
    xc = zeros(eltype(b), length(b))
    SV._amg_solve_mode!(w, w.hierarchy, s, outer, SV.outer_operator(w.hierarchy, A), b, xc; itmax=100, atol=1e-8, rtol=1e-8)
    @test norm(b - Array(parent(A)) * xc) / norm(b) < 1e-8
end

# NEW SECTION: mixed precision (Float32 coarse storage), FP64 outer correction reaches FP64 tolerance
for outer in (Cg(), Richardson())
    sref = AMG(SmoothedAggregation())
    s32  = SV._with_storage(sref, Float32)
    runamg(s) = begin
        w = _workspace(s, b; solver=outer); w = SV.update!(w, A, s, config)
        x = zeros(eltype(b), length(b))
        SV._amg_solve_mode!(w, w.hierarchy, s, outer, SV.outer_operator(w.hierarchy, A), b, x; itmax=200, atol=1e-8, rtol=1e-8)
        (it=w.iterations, res=norm(b - Array(parent(A)) * x) / norm(b),
         store=eltype(SV._nzval(w.hierarchy.levels[1].A)), mixed=SV._amg_mixed_precision(w.hierarchy))
    end
    rref = runamg(sref); r32 = runamg(s32)
    @test rref.store === Float64 && !rref.mixed
    @test r32.store === Float32 && r32.mixed
    @test r32.res < 1e-8
    # Cg: fixed SPD preconditioner, exact iteration parity; Richardson: FP64 IR around an FP32 cycle
    if outer isa Cg
        @test r32.it == rref.it
    else
        @test abs(r32.it - rref.it) <= max(2, rref.it ÷ 5)
    end
end

# Pairwise: Float32 coarse levels below a full-precision finest level
for outer in (Cg(), Richardson())
    s = SV._with_storage(AMG(Pairwise(); coarsest=2), Float32)
    w = SV.update!(_workspace(s, b; solver=outer), A, s, config)
    @test eltype(w.hierarchy.coarse_x) === Float32
    xm = zeros(eltype(b), length(b))
    SV._amg_solve_mode!(w, w.hierarchy, s, outer, SV.outer_operator(w.hierarchy, A), b, xm; itmax=200, atol=1e-8, rtol=1e-8)
    @test norm(b - Array(parent(A)) * xm) / norm(b) < 1e-8
end

# FP32 refresh path: a changed matrix (same pattern) refreshes the FP32 hierarchy in place and still converges
let s = SV._with_storage(AMG(SmoothedAggregation()), Float32)
    w = _workspace(s, b); w = SV.update!(w, A, s, config)
    h_before = w.hierarchy
    A2 = deepcopy(A); SV._nzval(A2) .*= 1.07
    w = SV.update!(w, A2, s, config)
    @test w.refresh_count >= 1
    @test w.hierarchy === h_before
    x = zeros(eltype(b), length(b))
    SV._amg_solve_mode!(w, w.hierarchy, s, Cg(), A2, b, x; itmax=200, atol=1e-8, rtol=1e-8)
    @test norm(b - Array(parent(A2)) * x) / norm(b) < 1e-7
end

# NEW SECTION: Float32-valued matrix (Int32+Float32 and Int64+Float32)
# SuiteSparse factorizes only Float64, so the host coarse direct solve runs in Float64 while the
# hierarchy stays Float32; storage is clamped to the working precision (no upcast)
function amg_test_matrix_i32(::Type{T}) where {T}
    i = Int32[1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
    j = Int32[1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5]
    v = T[300, -100, -100, 200, -100, -100, 200, -100, -100, 200, -100, -100, 300]
    return SparseXCSR(sparsecsr(i, j, v, 5, 5)), T[200 * 500, 0, 0, 0, 200 * 100]
end
# every index type, storage and outer iteration appears twice; the full cross product only repeats these paths
for (makeA, storage, outer) in ((amg_test_matrix, Float32, Cg()), (amg_test_matrix, Float64, Richardson()),
                                (amg_test_matrix_i32, Float32, Richardson()), (amg_test_matrix_i32, Float64, Cg()))
    Af, bf = makeA(Float32)
    @test eltype(SV._nzval(Af)) === Float32
    s = SV._with_storage(AMG(SmoothedAggregation()), storage)
    w = _workspace(s, bf; solver=outer)
    w = SV.update!(w, Af, s, config)
    @test eltype(SV._nzval(w.hierarchy.levels[1].A)) === Float32
    @test !SV._amg_mixed_precision(w.hierarchy)
    @test eltype(w.hierarchy.coarse_cpu.Acsc) === Float64
    xf = zeros(Float32, length(bf))
    SV._amg_solve_mode!(w, w.hierarchy, s, outer, w.hierarchy.levels[1].A, bf, xf; itmax=200, atol=1f-6, rtol=1f-6)
    @test norm(bf - Array(parent(Af)) * xf) / norm(bf) < 1e-4
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
        ws_gpu = _workspace(setup.preconditioner, b_gpu; solver=Richardson())
        ws_gpu = SV.update!(ws_gpu, A_gpu, setup.preconditioner, config_gpu)
        @test SV._nzval(ws_gpu.hierarchy.levels[1].A) isa CuArray
        @test ws_gpu.hierarchy.coarse_cpu.rhs isa Vector

        i2, j2, v2 = findnz(parent(A2))
        A2_gpu = XCALibre.ModelFramework._build_A(backend_gpu, i2, j2, v2, size(A2, 1))
        ws_gpu = SV.update!(ws_gpu, A2_gpu, setup.preconditioner, config_gpu)
        @test Array(SV._nzval(ws_gpu.hierarchy.levels[1].A)) == Array(XCALibre.ModelFramework._nzval(A2))

        SV.amg_solve!(ws_gpu, ws_gpu.hierarchy, setup.preconditioner, ws_gpu.hierarchy.levels[1].A, b_gpu, x_gpu; itmax=20, atol=1e-8, rtol=1e-8)
        @test norm(Array(b_gpu) - Array(parent(A2)) * Array(x_gpu)) / norm(Array(b_gpu)) < 1e-6

        solver_cg_gpu = AMG(SmoothedAggregation())
        ws_cg_gpu = _workspace(solver_cg_gpu, b_gpu)
        ws_cg_gpu = SV.update!(ws_cg_gpu, A_gpu, solver_cg_gpu, config_gpu)
        x_cg_gpu = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
        SV.amg_cg_solve!(ws_cg_gpu, ws_cg_gpu.hierarchy, solver_cg_gpu, ws_cg_gpu.hierarchy.levels[1].A, b_gpu, x_cg_gpu; itmax=20, atol=1e-8, rtol=1e-8)
        @test norm(Array(b_gpu) - Array(parent(A)) * Array(x_cg_gpu)) / norm(Array(b_gpu)) < 1e-6

        # mixed precision on device: FP32 cuSPARSE hierarchy + FP64 outer (raw A) reaches FP64 tol
        for outer_gpu in (Cg(), Richardson())
            s32g = SV._with_storage(AMG(SmoothedAggregation()), Float32)
            w32g = _workspace(s32g, b_gpu; solver=outer_gpu)
            w32g = SV.update!(w32g, A_gpu, s32g, config_gpu)
            @test SV._amg_mixed_precision(w32g.hierarchy)
            @test SV._nzval(w32g.hierarchy.levels[1].A) isa CuArray{Float32}
            x32g = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
            SV._amg_solve_mode!(w32g, w32g.hierarchy, s32g, outer_gpu, A_gpu, b_gpu, x32g; itmax=50, atol=1e-8, rtol=1e-8)
            @test norm(Array(b_gpu) - Array(parent(A)) * Array(x32g)) / norm(Array(b_gpu)) < 1e-6
            # FP32 device refresh: FP64->FP32 broadcast, device FP32 RAP and mixed coarse rebuild
            w32g = SV.update!(w32g, A2_gpu, s32g, config_gpu)
            x32g2 = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
            SV._amg_solve_mode!(w32g, w32g.hierarchy, s32g, outer_gpu, A2_gpu, b_gpu, x32g2; itmax=50, atol=1e-8, rtol=1e-8)
            @test norm(Array(b_gpu) - Array(parent(A2)) * Array(x32g2)) / norm(Array(b_gpu)) < 1e-6
        end

        # fused split-precision device refresh: the coarse RAP gather adds across precisions (F64 finest ->
        # F32 coarse); needs a matrix large enough to coarsen at max_coarse=64
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
            let rp = SV._rowptr(g2), cv = SV._colval(g2), nz = SV._nzval(g2)
                for r in 1:SV._m(g2), p in rp[r]:(rp[r+1]-1)
                    nz[p] *= (cv[p] == r ? 1.25 : 0.85)
                end
            end
            rm = SV.validate_refresh(g1, g2, 1, backend_gpu; max_coarse=64, coarse_storage=Float32)
            @test rm.relerr < 1e-5
            cg64 = SV.validate_refresh_convergence(g1, g2, 1, backend_gpu; max_coarse=64, coarse_storage=Float64)
            cg32 = SV.validate_refresh_convergence(g1, g2, 1, backend_gpu; max_coarse=64, coarse_storage=Float32)
            @test cg32.converged && cg32.iters <= cg64.iters + 1
        end

        solver_chebyshev_gpu = AMG(SmoothedAggregation(); smoother=Chebyshev())
        ws_chebyshev_gpu = _workspace(solver_chebyshev_gpu, b_gpu)
        ws_chebyshev_gpu = SV.update!(ws_chebyshev_gpu, A_gpu, solver_chebyshev_gpu, config_gpu)
        x_chebyshev_gpu = KernelAbstractions.zeros(backend_gpu, eltype(b), length(b))
        SV.amg_cg_solve!(ws_chebyshev_gpu, ws_chebyshev_gpu.hierarchy, solver_chebyshev_gpu,
            ws_chebyshev_gpu.hierarchy.levels[1].A, b_gpu, x_chebyshev_gpu; itmax=20, atol=1e-8, rtol=1e-8)
        @test norm(Array(b_gpu) - Array(parent(A)) * Array(x_chebyshev_gpu)) / norm(Array(b_gpu)) < 1e-6

        solver_gs_gpu = AMG(SmoothedAggregation(); smoother=GaussSeidel())
        err = try
            SV.setup_hierarchy(A_gpu, solver_gs_gpu, backend_gpu; log_diagnostics=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin("sequential CPU smoother", message)
        @test occursin("Jacobi()", message)
        @test occursin("Chebyshev()", message)

        # GPU fuse: Pairwise keeps aggregate lists; 0 and 1 fused levels agree in iterations (±1) and solution.
        # Screened Poisson keeps the weight=4/3 Jacobi smoother stable; pure Poisson makes the Pairwise cycle diverge.
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
            run_fuse(fuse; smoother=Jacobi(), coarse=Direct(), solver=Cg()) = begin
                s = AMG(Pairwise(1); smoother, coarsest=128, native=NativeAMG(; fuse, coarse))
                ws = _workspace(s, b_t3; solver)
                ws = SV.update!(ws, A_t3, s, config_gpu)
                x = KernelAbstractions.zeros(backend_gpu, Float64, np)
                SV._amg_solve_mode!(ws, ws.hierarchy, s, solver, A_t3, b_t3, x; itmax=200, atol=0.0, rtol=1e-8)
                (iters=ws.iterations, x=Array(x), mf=ws.hierarchy isa SV.MatrixFreeHierarchy, conv=ws.converged)
            end
            r0 = run_fuse(0); r1 = run_fuse(1)
            @test r0.mf && r1.mf
            @test abs(r1.iters - r0.iters) <= 1
            @test norm(r1.x - r0.x) / norm(r0.x) < 1e-8
            r2 = run_fuse(2)
            @test r2.mf
            @test norm(ones(np) - Ah_t3 * r2.x) / sqrt(np) < 1e-6
            # Chebyshev on fused levels, and the device coarse solves of the aggregate hierarchy
            for r ∈ (run_fuse(1; smoother=Chebyshev()), run_fuse(0; coarse=Smoothed(Chebyshev(degree=8))),
                    run_fuse(0; coarse=Smoothed(Jacobi(); sweeps=20)), run_fuse(0; coarse=Iterative(Cg()), solver=Richardson()))
                @test r.conv && norm(ones(np) - Ah_t3 * r.x) / sqrt(np) < 1e-6
            end
        end
    end
catch err
    @info "Skipping CUDA AMG test" exception=(err, catch_backtrace())
end
