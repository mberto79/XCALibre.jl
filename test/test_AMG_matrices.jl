# Correctness + loose regression suite over named structured matrices

using SparseArrays

# MATRIX BUILDERS

function poisson1d(n=50)
    i = vcat(1:n, 1:n-1, 2:n)
    j = vcat(1:n, 2:n, 1:n-1)
    v = vcat(fill(2.0, n), fill(-1.0, 2*(n-1)))
    SparseXCSR(sparsecsr(i, j, v, n, n)), ones(n)
end

function poisson2d(m=10)
    n = m * m
    rows, cols, vals = Int[], Int[], Float64[]
    for r in 1:m, c in 1:m
        k = (r-1)*m + c
        push!(rows, k); push!(cols, k); push!(vals, 4.0)
        if c > 1; push!(rows, k); push!(cols, k-1); push!(vals, -1.0); end
        if c < m; push!(rows, k); push!(cols, k+1); push!(vals, -1.0); end
        if r > 1; push!(rows, k); push!(cols, k-m); push!(vals, -1.0); end
        if r < m; push!(rows, k); push!(cols, k+m); push!(vals, -1.0); end
    end
    SparseXCSR(sparsecsr(rows, cols, vals, n, n)), ones(n)
end

function anisotropic1d(n=50)
    # graded coefficients: d[i] = 2+sin(i), symmetric off-diag; SPD, diag dominant
    d = [2.0 + sin(Float64(i)) for i in 1:n]
    off = [-0.5 - 0.3*cos(Float64(i)) for i in 1:n-1]  # |off| < 0.9 < min(d); weakly diag dominant, SPD
    i = vcat(1:n, 1:n-1, 2:n)
    j = vcat(1:n, 2:n, 1:n-1)
    v = vcat(d, off, off)
    SparseXCSR(sparsecsr(i, j, v, n, n)), ones(n)
end

# SHARED HELPERS

function _make_config()
    setup = SolverSetup(
        solver=AMG(),
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=1.0,
        rtol=1e-8,
        atol=1e-8
    )
    Configuration(
        solvers=(T=setup,),
        schemes=(T=Schemes(),),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
        hardware=Hardware(backend=CPU(), workgroup=64),
        boundaries=(T=(),)
    )
end

function _solve_amg(A, b, solver, config; itmax=500)
    ws = _workspace(solver, b)
    ws = XCALibre.Solve.update!(ws, A, solver, config)
    x = zeros(eltype(b), length(b))
    XCALibre.Solve._amg_solve_mode!(
        ws, ws.hierarchy, solver, solver.mode,
        ws.hierarchy.levels[1].A, b, x;
        itmax=itmax, atol=1e-8, rtol=1e-8
    )
    relres = norm(b - Array(parent(A)) * x) / norm(b)
    relres, ws
end

# TESTS

const _config = _make_config()

@testset "poisson1d" begin
    A, b = poisson1d()

    # A. CORRECTNESS
    for mode in (Cg(), AMGSolver()),
        coarsening in (SmoothAggregation(), RugeStuben(), Geometric()),
        smoother in (AMGJacobi(), AMGChebyshev(), AMGGaussSeidel(), AMGSOR(1.0))
        solver = AMG(mode=mode, coarsening=coarsening, smoother=smoother, max_coarse_rows=8)
        relres, _ = _solve_amg(A, b, solver, _config)
        @test relres < 1e-7
    end

    # B. REGRESSION — default solver: Cg + SmoothAggregation + AMGJacobi, max_coarse_rows=8
    reg_solver = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGJacobi(), max_coarse_rows=8)
    _, ws_reg = _solve_amg(A, b, reg_solver, _config)
    recorded_p1d = 8 # loose regression baseline (observed 8)
    @test ws_reg.iterations <= 2 * recorded_p1d + 20

    # C. OMEGA CONVENTION
    ws_omega = _workspace(reg_solver, b)
    ws_omega = XCALibre.Solve.update!(ws_omega, A, reg_solver, _config)
    level = ws_omega.hierarchy.levels[1]
    @test XCALibre.Solve._level_jacobi_omega(AMGJacobi(), level) ≈ 4 / (3 * level.lambda_max)
end

@testset "poisson2d" begin
    A, b = poisson2d()

    # A. CORRECTNESS
    for mode in (Cg(), AMGSolver()),
        coarsening in (SmoothAggregation(), RugeStuben(), Geometric()),
        smoother in (AMGJacobi(), AMGChebyshev(), AMGGaussSeidel(), AMGSOR(1.0))
        solver = AMG(mode=mode, coarsening=coarsening, smoother=smoother, max_coarse_rows=8)
        relres, _ = _solve_amg(A, b, solver, _config)
        @test relres < 1e-7
    end

    # B. REGRESSION
    reg_solver = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGJacobi(), max_coarse_rows=8)
    _, ws_reg = _solve_amg(A, b, reg_solver, _config)
    recorded_p2d = 8 # loose regression baseline (observed 8)
    @test ws_reg.iterations <= 2 * recorded_p2d + 20
end

@testset "anisotropic1d" begin
    A, b = anisotropic1d()

    # A. CORRECTNESS
    for mode in (Cg(), AMGSolver()),
        coarsening in (SmoothAggregation(), RugeStuben(), Geometric()),
        smoother in (AMGJacobi(), AMGChebyshev(), AMGGaussSeidel(), AMGSOR(1.0))
        solver = AMG(mode=mode, coarsening=coarsening, smoother=smoother, max_coarse_rows=8)
        relres, _ = _solve_amg(A, b, solver, _config)
        @test relres < 1e-7
    end

    # B. REGRESSION
    reg_solver = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGJacobi(), max_coarse_rows=8)
    _, ws_reg = _solve_amg(A, b, reg_solver, _config)
    recorded_ani1d = 5 # loose regression baseline (observed 5)
    @test ws_reg.iterations <= 2 * recorded_ani1d + 20
end
