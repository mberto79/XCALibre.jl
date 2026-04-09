# ─── AMG-internal smoothers ────────────────────────────────────────────────────
# These operate on MultigridLevel objects and use the KA kernels from AMG_1_kernels.jl.
# They are distinct from the pre-solver JacobiSmoother in Smoothers_jacobi.jl.

# ─── Damped Jacobi ────────────────────────────────────────────────────────────

"""
    amg_smooth!(level, n_sweeps, omega, backend, workgroup)

Apply `n_sweeps` of damped Jacobi smoothing on a multigrid level.
Uses `level.tmp` as the swap buffer — zero allocation per sweep.
"""
function amg_smooth!(level::MultigridLevel, n_sweeps::Int, omega,
                      backend, workgroup)
    for _ in 1:n_sweeps
        # x_new stored in level.tmp
        amg_jacobi_sweep!(level.tmp, level.x, level.Dinv,
                           level.A, level.b, omega, backend, workgroup)
        # swap x ↔ tmp
        level.x, level.tmp = level.tmp, level.x
    end
end

# ─── Chebyshev polynomial smoother ────────────────────────────────────────────
# Implements a degree-k Chebyshev iteration as in AMGX.
# Eigenvalue window [λ_min, λ_max] estimated from ρ: λ_max ≈ hi*ρ, λ_min ≈ lo*ρ

"""
    amg_smooth_chebyshev!(level, smoother, backend, workgroup)

Apply Chebyshev polynomial smoothing on `level` using parameters from `smoother::Chebyshev`.
"""
function amg_smooth_chebyshev!(level::MultigridLevel, smoother::Chebyshev,
                                 backend, workgroup)
    (; degree, lo, hi) = smoother
    rho = level.rho[]

    # Eigenvalue bounds
    λ_max = hi * rho
    λ_min = lo * rho

    d = (λ_max + λ_min) / 2   # centre
    c = (λ_max - λ_min) / 2   # radius

    # Chebyshev iteration (standard 3-term recurrence):
    #   p_0 = x
    #   r_0 = b - A x_0
    #   p_1 = x_0 + (1/d) * r_0
    #   ...
    # See Notay 2010 / AMGX documentation for the exact recurrence used here.

    # We use level.r for the residual and level.tmp as scratch.
    amg_residual!(level.r, level.A, level.x, level.b, backend, workgroup)

    # Initial correction z_0 = (1/d) * r
    alpha = one(eltype(level.x)) / d
    amg_axpy!(level.x, level.r, alpha, backend, workgroup)  # x += α r

    if degree == 1
        return
    end

    # Higher-degree recurrence
    # z_{k+1} = (2/d) * z_k * rho_k - z_{k-1}  (Chebyshev recurrence on correction)
    # We maintain the previous residual in tmp.
    amg_copy!(level.tmp, level.r, backend, workgroup)   # tmp = r_0

    rho_prev = one(eltype(level.x))
    for _ in 2:degree
        # Compute new residual r = b - A x
        amg_residual!(level.r, level.A, level.x, level.b, backend, workgroup)

        rho_new = one(eltype(level.x)) / (2*d/c - rho_prev)
        alpha_k = rho_new * rho_prev * 2 / c
        beta_k  = -rho_new

        # x = (1 + beta_k) * x_old_old  +  alpha_k * r  (using tmp to hold previous)
        # Update in place: x += alpha_k*r + beta_k*(x - x_prev) would need x_prev
        # Use: z = beta_k * z_prev + alpha_k * r  and then x += z
        # level.tmp stores the previous z
        # z_new = alpha_k * r + beta_k * z_prev
        amg_axpby!(level.tmp, level.r, alpha_k, beta_k, backend, workgroup)
        amg_axpy!(level.x, level.tmp, one(eltype(level.x)), backend, workgroup)

        rho_prev = rho_new
    end
end

# ─── Dispatch: apply smoother based on type ────────────────────────────────────

function apply_level_smoother!(level::MultigridLevel, n_sweeps::Int,
                                 opts::AMG, backend, workgroup)
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
