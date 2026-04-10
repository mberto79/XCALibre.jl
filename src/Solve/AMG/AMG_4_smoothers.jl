# ─── AMG-internal smoothers ────────────────────────────────────────────────────
# These operate on MultigridLevel objects and use the KA kernels from AMG_1_kernels.jl.
# They are distinct from the pre-solver JacobiSmoother in Smoothers_jacobi.jl.

# ─── Damped Jacobi (correction form) ─────────────────────────────────────────
# Classical:  x_new[i] = ω/a_ii*(b[i] - Σ_{j≠i} a_ij x[j]) + (1-ω)x[i]
# Equivalent: x += ω·D⁻¹·r  (r = b - Ax); no tmp-buffer swap needed.

"""
    amg_smooth!(level, n_sweeps, omega, backend, workgroup)

Apply `n_sweeps` of damped Jacobi smoothing using the fused in-place kernel:
`x[i] += ω * Dinv[i] * (b[i] - Ax[i])` per row. Zero allocation per sweep;
does not write `level.r`.
"""
function amg_smooth!(level, n_sweeps, omega, backend, workgroup)
    for _ in 1:n_sweeps
        amg_smooth_jacobi!(level.x, level.Dinv, level.A, level.b, omega, backend, workgroup)
    end
end

# ─── Chebyshev polynomial smoother ────────────────────────────────────────────
# Implements a degree-k Chebyshev iteration as in AMGX.
# Eigenvalue window [λ_min, λ_max] estimated from ρ: λ_max ≈ hi*ρ, λ_min ≈ lo*ρ

"""
    amg_smooth_chebyshev!(level, smoother, backend, workgroup)

Apply Chebyshev polynomial smoothing on `level` using parameters from `smoother::Chebyshev`.
"""
function amg_smooth_chebyshev!(level, smoother::Chebyshev, backend, workgroup)
    (; degree, lo, hi) = smoother
    rho = level.extras.rho

    # Eigenvalue bounds
    λ_max = hi * rho
    λ_min = lo * rho

    d = (λ_max + λ_min) / 2   # centre
    c = (λ_max - λ_min) / 2   # radius

    # Chebyshev iteration (standard 3-term recurrence)
    amg_residual!(level.r, level.A, level.x, level.b, backend, workgroup)

    # Initial correction z_0 = (1/d) * r
    alpha = one(eltype(level.x)) / d
    amg_axpy!(level.x, level.r, alpha, backend, workgroup)

    if degree == 1
        return
    end

    amg_copy!(level.tmp, level.r, backend, workgroup)

    rho_prev = one(eltype(level.x))
    for _ in 2:degree
        amg_residual!(level.r, level.A, level.x, level.b, backend, workgroup)

        rho_new   = one(eltype(level.x)) / (2*d/c - rho_prev)
        alpha_k   = rho_new * rho_prev * 2 / c
        beta_k    = -rho_new

        amg_axpby!(level.tmp, level.r, alpha_k, beta_k, backend, workgroup)
        amg_axpy!(level.x, level.tmp, one(eltype(level.x)), backend, workgroup)

        rho_prev = rho_new
    end
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
