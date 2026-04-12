# ─── AMG-internal smoothers ────────────────────────────────────────────────────
# These operate on MultigridLevel objects and use the KA kernels from AMG_1_kernels.jl.
# They are distinct from the pre-solver JacobiSmoother in Smoothers_jacobi.jl.

# ─── Damped Jacobi (synchronous, ping-pong) ───────────────────────────────────
# Uses the two-buffer _amg_jacobi_sweep! kernel so reads and writes never race.
# This is required for use as a PCG preconditioner: the preconditioner must be
# a fixed linear map (chaotic in-place Jacobi is non-deterministic, breaking
# CG's A-conjugacy). `level.tmp` is the second ping-pong buffer.

"""
    amg_smooth!(level, n_sweeps, omega, backend, workgroup)

Apply `n_sweeps` of synchronous damped Jacobi using ping-pong between
`level.x` and `level.tmp`. Ensures the preconditioner is deterministic and
symmetric — mandatory for correctness when AMG is a PCG preconditioner.
"""
function amg_smooth!(level, n_sweeps, omega, backend, workgroup)
    src = level.x
    dst = level.tmp
    for _ in 1:n_sweeps
        amg_jacobi_sweep!(dst, src, level.Dinv, level.A, level.b, omega, backend, workgroup)
        src, dst = dst, src
    end
    # After an odd number of sweeps the result ends in what started as level.tmp; copy back.
    src !== level.x && amg_copy!(level.x, src, backend, workgroup)
end

# ─── Chebyshev polynomial smoother ────────────────────────────────────────────
# Preconditioned Chebyshev targeting the operator D⁻¹A with eigenvalues in
# [λ_min, λ_max] (bounded by Gershgorin on D⁻¹A at setup time).
# All corrections apply D⁻¹ so the bounds and the iterated operator match.
# References: Saad "Iterative Methods for Sparse Linear Systems" §12.1;
#             Adams et al., "Parallel Multigrid Smoothing", §3.

"""
    amg_smooth_chebyshev!(level, smoother, backend, workgroup)

Apply a degree-k preconditioned Chebyshev iteration targeting D⁻¹A.
Eigenvalue window [λ_min, λ_max] ≈ [lo·ρ, hi·ρ] where ρ = ρ(D⁻¹A).
Corrections are scaled by D⁻¹ at every step — required for consistency with
the eigenvalue bounds computed by `_gershgorin_rho`.
"""
function amg_smooth_chebyshev!(level, smoother::Chebyshev, backend, workgroup)
    (; degree, lo, hi) = smoother
    rho = level.extras.rho
    Tv  = eltype(level.x)

    # Eigenvalue bounds for D⁻¹A
    λ_max = hi * rho
    λ_min = lo * rho

    d = (λ_max + λ_min) / 2   # centre
    c = (λ_max - λ_min) / 2   # radius

    # Step 0: r = b - Ax;  x += (1/d) * D⁻¹ r
    amg_residual!(level.r, level.A, level.x, level.b, backend, workgroup)
    alpha = one(Tv) / d
    amg_dinv_axpy!(level.x, level.Dinv, level.r, alpha, backend, workgroup)

    degree == 1 && return

    # p₀ = step taken in step 0 = (1/d) D⁻¹ r₀  (the actual update direction, not just D⁻¹ r)
    amg_dinv_axpby!(level.tmp, level.Dinv, level.r, alpha, zero(Tv), backend, workgroup)

    rho_prev = one(Tv)
    for _ in 2:degree
        amg_residual!(level.r, level.A, level.x, level.b, backend, workgroup)

        rho_new = one(Tv) / (2*d/c - rho_prev)
        alpha_k = rho_new * 2 / c          # ρ_k · (2/δ)
        beta_k  = rho_new * rho_prev       # ρ_k · ρ_{k-1}  (positive)

        # p_k = alpha_k · D⁻¹ r + beta_k · p_{k-1}  (Adams et al. 2003, eq. 3)
        amg_dinv_axpby!(level.tmp, level.Dinv, level.r, alpha_k, beta_k, backend, workgroup)
        amg_axpy!(level.x, level.tmp, one(Tv), backend, workgroup)

        rho_prev = rho_new
    end
end

# ─── L1-Jacobi smoother (synchronous ping-pong) ───────────────────────────────
# Uses the fused _amg_l1jacobi_sweep! kernel so the full residual is computed
# from a snapshot of x, then x is updated — no diagonal-exclusion artefact.
# Ping-pong ensures the preconditioner is a fixed symmetric linear map (valid for PCG).

function amg_smooth_l1jacobi!(level, n_sweeps, omega, backend, workgroup)
    src = level.x
    dst = level.tmp
    for _ in 1:n_sweeps
        amg_l1jacobi_sweep!(dst, src, level.Dinv, level.A, level.b, omega, backend, workgroup)
        src, dst = dst, src
    end
    src !== level.x && amg_copy!(level.x, src, backend, workgroup)
end

# ─── Dispatch: apply smoother based on type ────────────────────────────────────

function apply_level_smoother!(level, n_sweeps::Int, opts::AMG, backend, workgroup)
    _apply_level_smoother!(level, opts.smoother, n_sweeps, backend, workgroup)
end

function _apply_level_smoother!(level, smoother::JacobiSmoother, n_sweeps::Int,
                                  backend, workgroup)
    amg_smooth!(level, n_sweeps, smoother.omega, backend, workgroup)
end

function _apply_level_smoother!(level, smoother::Chebyshev, n_sweeps::Int,
                                  backend, workgroup)
    for _ in 1:n_sweeps
        amg_smooth_chebyshev!(level, smoother, backend, workgroup)
    end
end

function _apply_level_smoother!(level, smoother::L1Jacobi, n_sweeps::Int,
                                  backend, workgroup)
    amg_smooth_l1jacobi!(level, n_sweeps, smoother.omega, backend, workgroup)
end
