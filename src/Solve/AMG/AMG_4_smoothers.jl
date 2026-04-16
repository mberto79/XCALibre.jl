# ─── AMG-internal smoothers ────────────────────────────────────────────────────
# These operate on MultigridLevel objects and use the KA kernels from AMG_1_kernels.jl.
# They are distinct from the pre-solver JacobiSmoother in Smoothers_jacobi.jl.

# ─── Damped Jacobi (synchronous, ping-pong) ───────────────────────────────────
# Two-buffer swap ensures deterministic output — required for PCG A-conjugacy.

function amg_smooth!(level, n_sweeps, omega, backend, workgroup)
    Tv    = eltype(level.x)
    omega_t = Tv(omega)   # prevent Float64 from promoting Float32 kernel args
    src = level.x
    dst = level.tmp
    for _ in 1:n_sweeps
        amg_jacobi_sweep!(dst, src, level.Dinv, level.A, level.b, omega_t, backend, workgroup)
        src, dst = dst, src
    end
    # After odd sweeps, result is in level.tmp; copy back to level.x.
    src !== level.x && amg_copy!(level.x, src, backend, workgroup)
end

# ─── Chebyshev polynomial smoother ────────────────────────────────────────────
# Preconditioned for D⁻¹A; eigenvalue window [lo·ρ, hi·ρ]; all corrections scaled by D⁻¹.

function amg_smooth_chebyshev!(level, smoother::Chebyshev, backend, workgroup)
    (; degree) = smoother
    Tv  = eltype(level.x)

    # Convert to level float type to prevent promotion from Float64 literals.
    lo  = Tv(smoother.lo)
    hi  = Tv(smoother.hi)
    rho = Tv(level.extras.rho)
    λ_max = hi * rho
    λ_min = lo * rho
    d = (λ_max + λ_min) / 2   # centre
    c = (λ_max - λ_min) / 2   # radius

    # Step 0: r = b - Ax; x += (1/d) * D⁻¹ r
    amg_residual!(level.r, level.A, level.x, level.b, backend, workgroup)
    alpha = one(Tv) / d
    amg_dinv_axpy!(level.x, level.Dinv, level.r, alpha, backend, workgroup)

    degree == 1 && return
    # p₀ = (1/d) D⁻¹ r₀ (actual update direction for Chebyshev recursion).
    amg_dinv_axpby!(level.tmp, level.Dinv, level.r, alpha, zero(Tv), backend, workgroup)

    rho_prev = one(Tv)
    for _ in 2:degree
        amg_residual!(level.r, level.A, level.x, level.b, backend, workgroup)
        rho_new = one(Tv) / (2*d/c - rho_prev)
        alpha_k = rho_new * 2 / c
        beta_k  = rho_new * rho_prev
        # p_k = alpha_k · D⁻¹ r + beta_k · p_{k-1}
        amg_dinv_axpby!(level.tmp, level.Dinv, level.r, alpha_k, beta_k, backend, workgroup)
        amg_axpy!(level.x, level.tmp, one(Tv), backend, workgroup)

        rho_prev = rho_new
    end
end

# ─── L1-Jacobi smoother (synchronous ping-pong) ───────────────────────────────
# Fused kernel: full residual from snapshot, then update. Ping-pong for deterministic PCG.

function amg_smooth_l1jacobi!(level, n_sweeps, omega, backend, workgroup)
    Tv    = eltype(level.x)
    omega_t = Tv(omega)   # prevent Float64 from promoting Float32 kernel args
    src = level.x
    dst = level.tmp
    for _ in 1:n_sweeps
        amg_l1jacobi_sweep!(dst, src, level.Dinv, level.A, level.b, omega_t, backend, workgroup)
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

# Fine-level F32 smoother: GPU override in CUDAExt; CPU falls back to F64.
function amg_smooth_fine_f32!(level, n_sweeps, omega, backend, workgroup)
    amg_smooth!(level, n_sweeps, omega, backend, workgroup)
end

# Type-stable dispatch: use F32 path when Dinv_Tc is available (fine level, GPU only).
@inline _apply_fine_smoother!(fine, n_sweeps, opts, backend, workgroup) =
    _fine_smoother_dispatch!(fine, n_sweeps, opts, fine.extras.Dinv_Tc, backend, workgroup)

_fine_smoother_dispatch!(fine, n_sweeps, opts, ::Nothing, backend, workgroup) =
    apply_level_smoother!(fine, n_sweeps, opts, backend, workgroup)

_fine_smoother_dispatch!(fine, n_sweeps, opts, _Dinv_Tc, backend, workgroup) =
    amg_smooth_fine_f32!(fine, n_sweeps, opts.smoother.omega, backend, workgroup)
