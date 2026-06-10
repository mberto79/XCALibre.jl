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

# Greenfield macro-aggregate layout (Phase 3): permutation + value-map reproduce the operator
@testset "macro_layout" begin
    A, _ = poisson2d(12)
    for ml in (1, 2, 3)
        lay = XCALibre.Solve.build_macro_layout(A, ml)
        @test lay.n_macro >= 1
        @test lay.agg_offsets[1] == 1
        @test lay.agg_offsets[end] == size(A, 1) + 1
        @test length(XCALibre.Solve._nzval(lay.A_perm)) == nnz(parent(A))
        @test XCALibre.Solve.macro_layout_spmv_error(A, lay) < 1e-12
    end
end

@testset "fused index types" begin
    S = XCALibre.Solve
    @test S._amg_index_type(100) == Int32
    @test S._amg_index_type(typemax(Int32)) == Int32
    @test S._amg_index_type(Int64(typemax(Int32)) + 1) == Int64
    @test S._amg_local_index_type(64) == UInt8
    @test S._amg_local_index_type(255) == UInt8
    @test S._amg_local_index_type(256) == UInt16
    @test S._amg_fused_workgroup(5) == 8
    @test S._amg_fused_workgroup(16) == 16
end

@testset "fused 2grid correction" begin
    # Independent oracle (reference build_prolongation P0 + sparse RAP + diagonal coarse solve) must
    # match the fused matrix-free kernel — validates in-aggregate coeffs + transfer composition.
    A, _ = poisson2d(16)
    for ml in (1, 2, 3)
        res = XCALibre.Solve.fused_2grid_oracle_error(A, ml, CPU())
        @test res.agg_match
        @test res.relerr < 1e-12
    end
end

@testset "mf multilevel cycle (5b-B)" begin
    # Full-hierarchy matrix-free V-cycle (P/R erased at EVERY level) must match the independent host
    # oracle built from reference sparse P_l/A_l, across multiple coarsening depths.
    A, _ = poisson2d(24)
    for ml in (1, 2, 3), (pre, post) in ((2, 2), (1, 1), (0, 2))
        res = XCALibre.Solve.mf_ml_cycle_spike(A, ml, CPU(); pre=pre, post=post, max_coarse=64)
        @test res.relerr < 1e-12
        @test res.levels >= 2
        @test res.vram_saved_bytes > 0
    end
end

@testset "mf scale_correction (sc)" begin
    # GAMG scale_correction (sf=(r_l·c)/(c·Ac)) in the MF up-sweep must (1) match the host oracle
    # extended with the SAME sf to ~eps (sf arithmetic + sc buffer mgmt), across depths/sweeps, and
    # (2) materially improve the stationary V-cycle (sc on converges where plain-additive stalls).
    A, _ = poisson2d(24)
    for ml in (1, 2, 3), (pre, post) in ((2, 2), (1, 1))
        res = XCALibre.Solve.mf_ml_cycle_spike(A, ml, CPU(); pre=pre, post=post, max_coarse=64,
                                               scale_correction=true)
        @test res.relerr < 1e-12
    end
    Ac, _ = poisson2d(40)
    off = XCALibre.Solve.mf_ml_convergence(Ac, 2, CPU(); scale_correction=false, itmax=80)
    on  = XCALibre.Solve.mf_ml_convergence(Ac, 2, CPU(); scale_correction=true,  itmax=80)
    @test on.converged
    @test on.iters <= off.iters
    # fused_top>0 with sc is unsupported (matrix-free coarse levels) -> must error loudly
    @test_throws ErrorException XCALibre.Solve._build_mf_ml(A, 2, CPU(); max_coarse=64,
                                                            fused_top=1, scale_correction=true)
end

@testset "mf top-k matrix-free Galerkin (5b-C)" begin
    # Top-k coarse operators applied matrix-free (A_l = R-chain·A0·P-chain) must reproduce the
    # materialized cycle EXACTLY (same operator), validating the prolong/restrict chain composition.
    A, _ = poisson2d(24)
    for ml in (2, 3), k in (1, 2)
        res = XCALibre.Solve.mf_ml_topk_error(A, ml, CPU(), k; pre=2, post=2, max_coarse=64)
        @test res.relerr < 1e-12
        @test res.fused_top >= 1
        @test res.vram_erased_bytes > 0
    end
end

@testset "mf coarse refresh (G1)" begin
    # Frozen-sparsity device refresh (refresh A1->A2) must reproduce a full rebuild on A2 to ~eps at
    # every level + the coarsest, including the recomputed omega. A2 shares A1's sparsity, different
    # values — this tests the frozen pattern is adequate for changed values, not just the arithmetic.
    A1, _ = poisson2d(24)
    A2 = deepcopy(A1)
    let rp = XCALibre.Solve._rowptr(A2), cv = XCALibre.Solve._colval(A2), nz = XCALibre.Solve._nzval(A2)
        for i in 1:XCALibre.Solve._m(A2), p in rp[i]:(rp[i+1]-1)
            nz[p] *= (cv[p] == i ? 1.25 : 0.85)
        end
    end
    for ml in (1, 2, 3)
        res = XCALibre.Solve.mf_ml_refresh_error(A1, A2, ml, CPU(); max_coarse=64)
        @test res.relerr < 1e-12
        @test res.omega_relerr < 1e-10
        @test res.levels >= 2
    end
    # split precision (finest F64 / coarse F32): refresh must reproduce the F64 build to coarse-F32
    # accuracy and stay iteration-equivalent. (GPU-only atomic type mismatch is guarded in test_AMG.jl.)
    for ml in (1, 2)
        rm = XCALibre.Solve.mf_ml_refresh_error(A1, A2, ml, CPU(); max_coarse=64, coarse_storage=Float32)
        @test rm.relerr < 1e-5
        c64 = XCALibre.Solve.mf_ml_refresh_convergence(A1, A2, ml, CPU(); max_coarse=64, coarse_storage=Float64)
        c32 = XCALibre.Solve.mf_ml_refresh_convergence(A1, A2, ml, CPU(); max_coarse=64, coarse_storage=Float32)
        @test c32.converged && c32.iters <= c64.iters + 1
    end
end

@testset "mf zero-alloc coarse solve (G2)" begin
    # The coarse handoff must allocate nothing that scales with coarse_n (was Array+backslash+adapt
    # every cycle). Both branches: device dense-inverse GEMV (small coarse) and reusable-buffer host
    # LU (large coarse). Per-cycle alloc must be constant; correctness preserved to ~eps either way.
    A, _ = poisson2d(24)
    for cmr in (512, 0)  # 512 -> GEMV branch (coarse_n=64<=512); 0 -> host-LU branch
        al = XCALibre.Solve.mf_ml_cycle_allocs(A, 2, CPU(); max_coarse=64, coarse_max_rows=cmr)
        @test al.constant
        @test al.branch == (cmr == 0 ? :host_lu : :gemv)
        sp = XCALibre.Solve.mf_ml_cycle_spike(A, 2, CPU(); max_coarse=64, coarse_max_rows=cmr)
        @test sp.relerr < 1e-12
    end
end

@testset "g3 block-smoother gate (G3)" begin
    # A single-launch fused top-zone kernel has no cross-workgroup sync, so for a preconditioner cycle
    # (x init 0, written only at exit) off-aggregate reads stay 0 in both smooths -> the fused smoother
    # is exactly aggregate-block-Jacobi. This gate measured the CG-iteration penalty on CPU and KILLED
    # the kernel (2.6-2.8x iters on poisson, divergence on F1). The test locks the finding: the
    # materialized baseline (depth 0) converges and block-smoothing the top zone always costs iters.
    A, _ = poisson2d(20)
    res = XCALibre.Solve.g3_block_smoother_gate(A, 2; max_coarse=64, max_depth=2)
    @test res.ratios[1] == 1.0
    @test res.finals[1] < 1e-6
    @test all(res.ratios[2:end] .> 1.0)
end

@testset "fused 2grid cycle (5b)" begin
    # Matrix-free 2-grid V-cycle (Jacobi pre/post + agg restrict/prolong + coupled RAP coarse solve)
    # must match the independent host oracle built from reference sparse P0/R — validates 1/sqrt(w)
    # consistency across restrict, prolong, and the materialized coarse operator.
    A, _ = poisson2d(20)
    for ml in (1, 2, 3), (pre, post) in ((1, 1), (2, 1), (0, 2))
        res = XCALibre.Solve.fused_2grid_cycle_spike(A, ml, CPU(); pre=pre, post=post)
        @test res.relerr < 1e-12
        @test res.vram_saved_bytes > 0
    end
end
