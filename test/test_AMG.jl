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
    SparseXCSR(sparsecsr(i, j, v, 5, 5)), T[200 * 500, 0, 0, 0, 200 * 100]
end

A, b = amg_test_matrix()
solver = AMG(mode=:solver, coarsening=SmoothAggregation(), smoother=AMGChebyshev())
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
@test setup.solver.presweeps == 1
@test setup.solver.postsweeps == 1
@test setup.solver.cycle == :V

solver_custom_sweeps = AMG(smoothing_steps=4)
@test solver_custom_sweeps.presweeps == 4
@test solver_custom_sweeps.postsweeps == 4

solver_explicit_sweeps = AMG(smoothing_steps=4, presweeps=2, postsweeps=3)
@test solver_explicit_sweeps.presweeps == 2
@test solver_explicit_sweeps.postsweeps == 3

solver_w = AMG(cycle=:W, adaptive_rebuild_factor=0.6)
@test solver_w.cycle == :W
@test solver_w.adaptive_rebuild_factor == 0.6

solver_refresh = AMG(coarse_refresh_interval=3, numeric_refresh_rtol=0.2)
@test solver_refresh.coarse_refresh_interval == 3
@test solver_refresh.numeric_refresh_rtol == 0.2

config = Configuration(
    solvers=(T=setup,),
    schemes=(T=Schemes(),),
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    hardware=Hardware(backend=CPU(), workgroup=64),
    boundaries=(T=(),)
)

ws = _workspace(setup.solver, b)
@test ws.solution !== ws.q
ws = XCALibre.Solve.update!(ws, A, setup.solver, config)
@test ws.hierarchy isa XCALibre.Solve.AMGHierarchy
@test ws.timing.build_calls == 1
@test ws.timing.last_update_action == :build
@test length(ws.hierarchy.levels) >= 1
@test ws.hierarchy.operator_complexity >= 1
@test ws.hierarchy.grid_complexity >= 1
if length(ws.hierarchy.levels) > 1
    P = ws.hierarchy.levels[1].P
    @test P !== nothing
    @test nnz(P) > size(P, 1)
end

solver_rs = AMG(mode=:solver, coarsening=RugeStuben(), smoother=AMGJacobi())
ws_rs = _workspace(solver_rs, b)
ws_rs = XCALibre.Solve.update!(ws_rs, A, solver_rs, config)
@test length(ws_rs.hierarchy.levels) >= 1
if length(ws_rs.hierarchy.levels) > 1
    P_rs = Matrix(ws_rs.hierarchy.levels[1].P)
    @test any(x -> x != 1.0, P_rs)
end

candidate = [1.0, 2.0, 3.0, 4.0, 5.0]
solver_sa_candidate = AMG(
    mode=:solver,
    coarsening=SmoothAggregation(near_nullspace=candidate),
    smoother=AMGJacobi()
)
ws_sa_candidate = _workspace(solver_sa_candidate, b)
ws_sa_candidate = XCALibre.Solve.update!(ws_sa_candidate, A, solver_sa_candidate, config)
if length(ws_sa_candidate.hierarchy.levels) > 1
    P_sa = Matrix(ws_sa_candidate.hierarchy.levels[1].P)
    @test maximum(abs, P_sa[:, 1]) != minimum(abs, P_sa[:, 1])
end

A2, _ = amg_test_matrix()
XCALibre.ModelFramework._nzval(A2) .= XCALibre.ModelFramework._nzval(A) .* 1.1
levels_before = length(ws.hierarchy.levels)
finest_before = copy(XCALibre.ModelFramework._nzval(ws.hierarchy.levels[1].A))
coarse_before = length(ws.hierarchy.levels) > 1 ? copy(XCALibre.ModelFramework._nzval(ws.hierarchy.levels[2].A)) : nothing
ws2 = XCALibre.Solve.update!(ws, A2, setup.solver, config)
@test length(ws2.hierarchy.levels) == levels_before
@test XCALibre.ModelFramework._nzval(ws2.hierarchy.levels[1].A) != finest_before
if length(ws2.hierarchy.levels) > 1
    @test ws2.timing.refresh_calls == 1
    @test ws2.timing.last_update_action == :refresh
    @test XCALibre.ModelFramework._nzval(ws2.hierarchy.levels[2].A) != coarse_before
else
    @test ws2.timing.finest_refresh_calls == 1
    @test ws2.timing.last_update_action == :finest_refresh
end

solver_skip_refresh = AMG(
    mode=:solver,
    coarsening=SmoothAggregation(),
    smoother=AMGJacobi(),
    coarse_refresh_interval=10,
    numeric_refresh_rtol=0.2
)
ws_skip = _workspace(solver_skip_refresh, b)
ws_skip = XCALibre.Solve.update!(ws_skip, A, solver_skip_refresh, config)
coarse_skip_before = length(ws_skip.hierarchy.levels) > 1 ? copy(XCALibre.ModelFramework._nzval(ws_skip.hierarchy.levels[2].A)) : nothing
A_small, _ = amg_test_matrix()
XCALibre.ModelFramework._nzval(A_small) .= XCALibre.ModelFramework._nzval(A) .* 1.01
ws_skip = XCALibre.Solve.update!(ws_skip, A_small, solver_skip_refresh, config)
@test ws_skip.hierarchy.reuse_steps == 1
@test ws_skip.timing.finest_refresh_calls == 1
@test ws_skip.timing.last_update_action == :finest_refresh
if length(ws_skip.hierarchy.levels) > 1
    @test XCALibre.ModelFramework._nzval(ws_skip.hierarchy.levels[2].A) == coarse_skip_before
end

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
XCALibre.Solve._apply_level_smoother!(AMGJacobi(), ws.hierarchy.levels[1], b, 3)
rj = norm(b - Array(parent(A)) * ws.hierarchy.levels[1].x)
@test rj < r0

fill!(ws.hierarchy.levels[1].x, 0)
XCALibre.Solve._apply_level_smoother!(AMGL1Jacobi(), ws.hierarchy.levels[1], b, 3)
rl1 = norm(b - Array(parent(A)) * ws.hierarchy.levels[1].x)
@test rl1 < r0

fill!(ws.hierarchy.levels[1].x, 0)
XCALibre.Solve._apply_level_smoother!(AMGSymmetricGaussSeidel(), ws.hierarchy.levels[1], b, 1)
rsgs = norm(b - Array(parent(A)) * ws.hierarchy.levels[1].x)
@test rsgs < r0

fill!(ws.hierarchy.levels[1].x, 0)
XCALibre.Solve._apply_level_smoother!(AMGChebyshev(), ws.hierarchy.levels[1], b, 2)
rc = norm(b - Array(parent(A)) * ws.hierarchy.levels[1].x)
@test rc < r0

x = zeros(eltype(b), length(b))
ws_solve = _workspace(setup.solver, b)
ws_solve = XCALibre.Solve.update!(ws_solve, A, setup.solver, config)
XCALibre.Solve.amg_solve!(ws_solve, ws_solve.hierarchy, setup.solver, A, b, x; itmax=100, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x) / norm(b) < 1e-8
@test ws_solve.hierarchy.last_cycle_factor >= 0
@test ws_solve.timing.apply_calls == ws_solve.iterations
@test ws_solve.timing.apply_time_s >= 0
@test length(ws_solve.residual_history) == ws_solve.iterations + 1
@test ws_solve.residual_history[end] <= ws_solve.residual_history[1]

solver_w_solve = AMG(
    mode=:solver,
    cycle=:W,
    coarsening=SmoothAggregation(),
    smoother=AMGJacobi(),
    max_levels=8,
    smoothing_steps=3
)
ws_w = _workspace(solver_w_solve, b)
ws_w = XCALibre.Solve.update!(ws_w, A, solver_w_solve, config)
x_w = zeros(eltype(b), length(b))
XCALibre.Solve.amg_solve!(ws_w, ws_w.hierarchy, solver_w_solve, A, b, x_w; itmax=40, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_w) / norm(b) < 1e-8

solver_cg = AMG(mode=:cg, coarsening=SmoothAggregation(), smoother=AMGJacobi())
ws_cg = _workspace(solver_cg, b)
ws_cg = XCALibre.Solve.update!(ws_cg, A, solver_cg, config)
@test ws_cg.hierarchy.is_symmetric
x_cg = zeros(eltype(b), length(b))
XCALibre.Solve.amg_cg_solve!(ws_cg, ws_cg.hierarchy, solver_cg, A, b, x_cg; itmax=50, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_cg) / norm(b) < 1e-8
@test ws_cg.timing.apply_calls >= ws_cg.iterations
@test ws_cg.timing.apply_time_s >= 0
@test length(ws_cg.residual_history) == ws_cg.iterations + 1
@test ws_cg.residual_history[end] <= ws_cg.residual_history[1]

A_bad = SparseXCSR(sparsecsr([1, 1, 2], [1, 2, 2], [2.0, 1.0, 3.0], 2, 2))
b_bad = [1.0, 1.0]
ws_bad = _workspace(solver_cg, b_bad)
ws_bad = XCALibre.Solve.update!(ws_bad, A_bad, solver_cg, config)
@test !ws_bad.hierarchy.is_symmetric
@test_throws ArgumentError XCALibre.Solve.amg_cg_solve!(ws_bad, ws_bad.hierarchy, solver_cg, A_bad, b_bad, zeros(2); itmax=10, atol=1e-8, rtol=1e-8)

solver_rebuild = AMG(
    mode=:solver,
    coarsening=SmoothAggregation(),
    smoother=AMGJacobi(),
    adaptive_rebuild_factor=0.0
)
ws_rebuild = _workspace(solver_rebuild, b)
ws_rebuild = XCALibre.Solve.update!(ws_rebuild, A, solver_rebuild, config)
x_rebuild = zeros(eltype(b), length(b))
XCALibre.Solve.amg_solve!(ws_rebuild, ws_rebuild.hierarchy, solver_rebuild, A, b, x_rebuild; itmax=1, atol=0.0, rtol=0.0)
@test ws_rebuild.hierarchy.last_cycle_factor >= 0
@test !ws_rebuild.hierarchy.force_rebuild
old_hierarchy = ws_rebuild.hierarchy
ws_rebuild = XCALibre.Solve.update!(ws_rebuild, A, solver_rebuild, config)
@test ws_rebuild.hierarchy === old_hierarchy
@test ws_rebuild.hierarchy.reuse_steps == 1
XCALibre.Solve.amg_solve!(ws_rebuild, ws_rebuild.hierarchy, solver_rebuild, A, b, x_rebuild; itmax=1, atol=0.0, rtol=0.0)
@test ws_rebuild.hierarchy.force_rebuild
old_hierarchy = ws_rebuild.hierarchy
ws_rebuild = XCALibre.Solve.update!(ws_rebuild, A, solver_rebuild, config)
@test ws_rebuild.hierarchy !== old_hierarchy

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

        i2, j2, v2 = findnz(parent(A2))
        A2_gpu = XCALibre.ModelFramework._build_A(backend_gpu, i2, j2, v2, size(A2, 1))
        ws_gpu = XCALibre.Solve.update!(ws_gpu, A2_gpu, setup.solver, config_gpu)
        @test Array(XCALibre.ModelFramework._nzval(ws_gpu.hierarchy.levels[1].A)) == Array(XCALibre.ModelFramework._nzval(A2))
        if length(ws_gpu.hierarchy.levels) > 1
            @test !isnothing(ws_gpu.hierarchy.levels[end].coarse_solver)
        end

        XCALibre.Solve.amg_solve!(ws_gpu, ws_gpu.hierarchy, setup.solver, A2_gpu, b_gpu, x_gpu; itmax=20, atol=1e-8, rtol=1e-8)
        @test norm(Array(b_gpu) - Array(parent(A2)) * Array(x_gpu)) / norm(Array(b_gpu)) < 1e-6
        @test ws_gpu.hierarchy.cpu_workspace !== nothing
    end
catch err
    @info "Skipping CUDA AMG test" exception=(err, catch_backtrace())
end
