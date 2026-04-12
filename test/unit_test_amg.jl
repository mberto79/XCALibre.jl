using XCALibre
using LinearAlgebra
using SparseMatricesCSR
using KernelAbstractions
using Test

import XCALibre.Solve: amg_setup!, update!, apply_level_smoother!, run_cycle!,
                       amg_residual!, _workspace, _amg_pcg_solve!

# ── Shared fixtures ────────────────────────────────────────────────────────────

# n×n interior 5-point Poisson matrix (SPD, near-isotropic FVM Laplacian).
function _build_poisson(n)
    N = n * n
    rows = Int[]; cols = Int[]; vals = Float64[]
    for i in 1:n, j in 1:n
        id = (i-1)*n + j
        push!(rows, id); push!(cols, id); push!(vals, 4.0)
        if i > 1; push!(rows,id); push!(cols,(i-2)*n+j); push!(vals,-1.0); end
        if i < n; push!(rows,id); push!(cols,i*n+j);     push!(vals,-1.0); end
        if j > 1; push!(rows,id); push!(cols,id-1);      push!(vals,-1.0); end
        if j < n; push!(rows,id); push!(cols,id+1);      push!(vals,-1.0); end
    end
    A_csr = sparsecsr(rows, cols, vals, N, N)
    return XCALibre.Multithread.SparseXCSR(A_csr)
end

const _amg_backend   = CPU()
const _amg_workgroup = 64

# Build a fully-initialised AMGWorkspace for a given smoother.
# Returns (ws, b, x_true) where b = A * x_true.
function _make_ws(smoother; n=8, strength=0.0, coarsening=:SA)
    N      = n * n
    A      = _build_poisson(n)
    x_true = rand(Float64, N)
    b      = zeros(Float64, N)
    mul!(b, A, x_true)

    opts = AMG(
        smoother      = smoother,
        cycle         = VCycle(),
        max_levels    = 8,
        coarsest_size = 8,
        pre_sweeps    = 2,
        post_sweeps   = 2,
        strength      = strength,
        coarsening    = coarsening,
    )
    ws = _workspace(opts, A, b)
    amg_setup!(ws, A, _amg_backend, _amg_workgroup)
    ws, b, x_true
end

# Apply the level-1 smoother (no coarse correction) for n_outer iterations;
# return the final relative residual ‖r‖/‖b‖.
function _smoother_converge_test(smoother; n=8, n_outer=300, strength=0.0)
    ws, b, _ = _make_ws(smoother; n=n, strength=strength)
    L1 = ws.levels[1]
    L1.b .= b
    L1.x .= 0.0

    r0 = norm(b)
    for _ in 1:n_outer
        apply_level_smoother!(L1, 1, ws.opts, _amg_backend, _amg_workgroup)
    end

    amg_residual!(L1.r, L1.A, L1.x, L1.b, _amg_backend, _amg_workgroup)
    norm(L1.r) / r0
end

# Run the full AMG V-cycle for up to max_iter cycles; return final relative residual.
function _amg_vcycle_converge_test(smoother; n=8, max_iter=60, strength=0.0, coarsening=:SA)
    ws, b, _ = _make_ws(smoother; n=n, strength=strength, coarsening=coarsening)
    L1 = ws.levels[1]
    L1.b .= b
    L1.x .= 0.0

    r0 = norm(b)
    for _ in 1:max_iter
        run_cycle!(ws.levels, ws.opts, ws.opts.cycle, _amg_backend, _amg_workgroup)
        amg_residual!(L1.r, L1.A, L1.x, L1.b, _amg_backend, _amg_workgroup)
        norm(L1.r) / r0 < 1e-10 && break
    end
    amg_residual!(L1.r, L1.A, L1.x, L1.b, _amg_backend, _amg_workgroup)
    norm(L1.r) / r0
end

# ── Tests ──────────────────────────────────────────────────────────────────────

@testset "AMG unit tests" begin

    # ── Hierarchy build and update ─────────────────────────────────────────────
    @testset "hierarchy build and update!" begin
        ws, b, _ = _make_ws(JacobiSmoother(2, 2/3, zeros(0)))
        @test length(ws.levels) > 1
        @test isconcretetype(eltype(ws.levels))

        A_new    = _build_poisson(8)
        n_before = length(ws.levels)
        update!(ws, A_new, _amg_backend, _amg_workgroup)
        @test length(ws.levels) == n_before   # update! must not change level count

        # lazy build: fresh workspace must trigger amg_setup! on first update!
        opts2 = AMG(smoother=JacobiSmoother(2, 2/3, zeros(0)))
        ws2   = _workspace(opts2, A_new, b)
        @test isempty(ws2.levels)
        update!(ws2, A_new, _amg_backend, _amg_workgroup)
        @test !isempty(ws2.levels)
    end

    # ── Damped Jacobi smoother ─────────────────────────────────────────────────
    @testset "JacobiSmoother (damped Jacobi)" begin
        smoother = JacobiSmoother(2, 2/3, zeros(0))

        # One sweep from zero must reduce the residual
        ws, b, _ = _make_ws(smoother)
        L1 = ws.levels[1]; L1.b .= b; L1.x .= 0.0
        r_before = norm(b)
        apply_level_smoother!(L1, 1, ws.opts, _amg_backend, _amg_workgroup)
        amg_residual!(L1.r, L1.A, L1.x, L1.b, _amg_backend, _amg_workgroup)
        @test norm(L1.r) < r_before

        # Standalone smoother: converges to correct solution after 300 sweeps
        rel = _smoother_converge_test(smoother; n=8, n_outer=300)
        @test rel < 1e-4

        # Full AMG V-cycle convergence
        rel_amg = _amg_vcycle_converge_test(smoother)
        @test rel_amg < 1e-8
    end

    # ── Chebyshev smoother ─────────────────────────────────────────────────────
    @testset "Chebyshev smoother" begin
        smoother = Chebyshev(degree=2, lo=0.3, hi=1.1)

        # One application from zero must reduce the residual
        ws, b, _ = _make_ws(smoother)
        L1 = ws.levels[1]; L1.b .= b; L1.x .= 0.0
        r_before = norm(b)
        apply_level_smoother!(L1, 1, ws.opts, _amg_backend, _amg_workgroup)
        amg_residual!(L1.r, L1.A, L1.x, L1.b, _amg_backend, _amg_workgroup)
        @test norm(L1.r) < r_before

        # Standalone smoother: converges to correct solution after 300 applications
        rel = _smoother_converge_test(smoother; n=8, n_outer=300)
        @test rel < 1e-4

        # degree=3 should not be significantly worse than degree=2 per call
        sm3  = Chebyshev(degree=3, lo=0.3, hi=1.1)
        rel3 = _smoother_converge_test(sm3;      n=8, n_outer=200)
        rel2 = _smoother_converge_test(smoother; n=8, n_outer=200)
        @test rel3 <= rel2 * 10

        # Full AMG V-cycle convergence
        rel_amg = _amg_vcycle_converge_test(smoother)
        @test rel_amg < 1e-8

        # Verify the recurrence direction is additive (beta_k > 0 fix):
        # successive applications must strictly decrease the residual — not oscillate.
        ws2, b2, _ = _make_ws(smoother)
        L2 = ws2.levels[1]; L2.b .= b2; L2.x .= 0.0
        apply_level_smoother!(L2, 1, ws2.opts, _amg_backend, _amg_workgroup)
        amg_residual!(L2.r, L2.A, L2.x, L2.b, _amg_backend, _amg_workgroup)
        r_after1 = norm(L2.r)
        apply_level_smoother!(L2, 1, ws2.opts, _amg_backend, _amg_workgroup)
        amg_residual!(L2.r, L2.A, L2.x, L2.b, _amg_backend, _amg_workgroup)
        r_after2 = norm(L2.r)
        @test r_after2 < r_after1   # no oscillation from wrong-sign beta
    end

    # ── L1-Jacobi smoother ─────────────────────────────────────────────────────
    @testset "L1Jacobi smoother" begin
        smoother = L1Jacobi(omega=1.0)

        # One sweep from zero must reduce the residual
        ws, b, _ = _make_ws(smoother)
        L1 = ws.levels[1]; L1.b .= b; L1.x .= 0.0
        r_before = norm(b)
        apply_level_smoother!(L1, 1, ws.opts, _amg_backend, _amg_workgroup)
        amg_residual!(L1.r, L1.A, L1.x, L1.b, _amg_backend, _amg_workgroup)
        @test norm(L1.r) < r_before

        # Standalone: converges to the CORRECT solution (pre-fix version converged to wrong point)
        rel = _smoother_converge_test(smoother; n=8, n_outer=300)
        @test rel < 1e-4

        # omega=4/3 compensates for the l1/diagonal ratio on isotropic FVM matrices
        sm_scaled  = L1Jacobi(omega=4/3)
        rel_scaled = _smoother_converge_test(sm_scaled; n=8, n_outer=150)
        rel_jacobi = _smoother_converge_test(JacobiSmoother(2, 2/3, zeros(0)); n=8, n_outer=150)
        @test rel_scaled < 0.5   # converging after 150 sweeps
        @test rel_jacobi < 0.5

        # Full AMG V-cycle convergence
        rel_amg = _amg_vcycle_converge_test(smoother)
        @test rel_amg < 1e-8
    end

    # ── PCG outer loop with each smoother ──────────────────────────────────────
    @testset "PCG solve (krylov=:cg) with each smoother" begin
        n   = 8
        N   = n * n
        A   = _build_poisson(n)
        x_t = rand(Float64, N)
        b   = zeros(Float64, N); mul!(b, A, x_t)

        for (label, sm) in [
            ("Jacobi",    JacobiSmoother(2, 2/3, zeros(0))),
            ("Chebyshev", Chebyshev(degree=2, lo=0.3, hi=1.1)),
            ("L1Jacobi",  L1Jacobi(omega=1.0)),
        ]
            @testset "$label" begin
                opts  = AMG(smoother=sm, krylov=:cg, max_levels=8,
                            coarsest_size=8, update_freq=1)
                ws    = _workspace(opts, A, b)
                amg_setup!(ws, A, _amg_backend, _amg_workgroup)

                x_out = zeros(Float64, N)
                _amg_pcg_solve!(ws, b, x_out, 40, 1e-10, 1e-10,
                                _amg_backend, _amg_workgroup)

                rel = norm(b - A * x_out) / norm(b)
                @test rel < 1e-8
            end
        end
    end

end # @testset "AMG unit tests"
