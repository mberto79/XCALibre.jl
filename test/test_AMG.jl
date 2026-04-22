using KernelAbstractions
using LinearAlgebra
using SparseMatricesCSR
using Test

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
@test setup.solver.presweeps == 10
@test setup.solver.postsweeps == 10

solver_custom_sweeps = AMG(smoothing_steps=4)
@test solver_custom_sweeps.presweeps == 4
@test solver_custom_sweeps.postsweeps == 4

solver_explicit_sweeps = AMG(smoothing_steps=4, presweeps=2, postsweeps=3)
@test solver_explicit_sweeps.presweeps == 2
@test solver_explicit_sweeps.postsweeps == 3

config = Configuration(
    solvers=(T=setup,),
    schemes=(T=Schemes(),),
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    hardware=Hardware(backend=CPU(), workgroup=64),
    boundaries=(T=(),)
)

ws = _workspace(setup.solver, b)
ws = XCALibre.Solve.update!(ws, A, setup.solver, config)
@test ws.hierarchy isa XCALibre.Solve.AMGHierarchy
@test length(ws.hierarchy.levels) >= 1
if length(ws.hierarchy.levels) > 1
    P = ws.hierarchy.levels[1].P
    @test P !== nothing
    @test nnz(P) > size(P, 1)
end

solver_rs = AMG(mode=:solver, coarsening=RugeStuben(), smoother=AMGJacobi())
ws_rs = _workspace(solver_rs, b)
ws_rs = XCALibre.Solve.update!(ws_rs, A, solver_rs, config)
@test length(ws_rs.hierarchy.levels) >= 1

A2, _ = amg_test_matrix()
XCALibre.ModelFramework._nzval(A2) .= XCALibre.ModelFramework._nzval(A) .* 1.1
levels_before = length(ws.hierarchy.levels)
ws2 = XCALibre.Solve.update!(ws, A2, setup.solver, config)
@test length(ws2.hierarchy.levels) == levels_before

x0 = zeros(eltype(b), length(b))
r0 = norm(b - Array(parent(A)) * x0)
fill!(ws.hierarchy.levels[1].x, 0)
XCALibre.Solve._apply_level_smoother!(AMGJacobi(), ws.hierarchy.levels[1], b, 3)
rj = norm(b - Array(parent(A)) * ws.hierarchy.levels[1].x)
@test rj < r0

fill!(ws.hierarchy.levels[1].x, 0)
XCALibre.Solve._apply_level_smoother!(AMGChebyshev(), ws.hierarchy.levels[1], b, 2)
rc = norm(b - Array(parent(A)) * ws.hierarchy.levels[1].x)
@test rc < r0

x = zeros(eltype(b), length(b))
ws_solve = _workspace(setup.solver, b)
ws_solve = XCALibre.Solve.update!(ws_solve, A, setup.solver, config)
XCALibre.Solve.amg_solve!(ws_solve, ws_solve.hierarchy, setup.solver, A, b, x; itmax=100, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x) / norm(b) < 1e-8

solver_cg = AMG(mode=:cg, coarsening=SmoothAggregation(), smoother=AMGJacobi())
ws_cg = _workspace(solver_cg, b)
ws_cg = XCALibre.Solve.update!(ws_cg, A, solver_cg, config)
x_cg = zeros(eltype(b), length(b))
XCALibre.Solve.amg_cg_solve!(ws_cg, ws_cg.hierarchy, solver_cg, A, b, x_cg; itmax=50, atol=1e-8, rtol=1e-8)
@test norm(b - Array(parent(A)) * x_cg) / norm(b) < 1e-8

A_bad = SparseXCSR(sparsecsr([1, 1, 2], [1, 2, 2], [2.0, 1.0, 3.0], 2, 2))
b_bad = [1.0, 1.0]
ws_bad = _workspace(solver_cg, b_bad)
ws_bad = XCALibre.Solve.update!(ws_bad, A_bad, solver_cg, config)
@test_throws ArgumentError XCALibre.Solve.amg_cg_solve!(ws_bad, ws_bad.hierarchy, solver_cg, A_bad, b_bad, zeros(2); itmax=10, atol=1e-8, rtol=1e-8)
