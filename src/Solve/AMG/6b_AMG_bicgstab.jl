# AMG-preconditioned BiCGStab: Krylov acceleration for the NON-SYMMETRIC operators
# `Cg()` refuses and `AMGSolver()` solves without acceleration (~5x the cycles).
# See "AMG solver" in the user guide for when to select it.

@kernel function _amg_bicg_p_kernel!(p, r, v, beta, omega)
    i = @index(Global)
    @inbounds p[i] = r[i] + beta * (p[i] - omega * v[i])
end

@kernel function _amg_bicg_s_kernel!(s, r, v, alpha)
    i = @index(Global)
    @inbounds s[i] = r[i] - alpha * v[i]
end

@kernel function _amg_bicg_update_kernel!(x, r, y, z, s, t, alpha, omega)
    i = @index(Global)
    @inbounds begin
        x[i] += alpha * y[i] + omega * z[i]
        r[i] = s[i] - omega * t[i]
    end
end

_bicg_p!(h, p, r, v, beta, omega) =
    (_launch_amg_kernel!(h, _amg_bicg_p_kernel!, length(p), p, r, v, beta, omega); p)

_bicg_s!(h, s, r, v, alpha) =
    (_launch_amg_kernel!(h, _amg_bicg_s_kernel!, length(s), s, r, v, alpha); s)

_bicg_update!(h, x, r, y, z, s, t, alpha, omega) =
    (_launch_amg_kernel!(h, _amg_bicg_update_kernel!, length(x), x, r, y, z, s, t,
                         alpha, omega); x)

# Right-preconditioned: `x` and `r` take the same `y`/`z` that were applied, so `r == b - A*x`
# survives a nonlinear `M^-1` (`scale_correction=true`). The recursion can still drift in
# finite precision, so convergence is confirmed on the true residual.
function amg_bicgstab_solve!(workspace::AMGWorkspace, hierarchy::AbstractAMGHierarchy,
                             solver::AMG, A, b, x; itmax, atol, rtol)
    T = eltype(x)
    r    = workspace.residual
    rhat = workspace.shadow
    p    = workspace.search
    v    = workspace.q
    y    = workspace.preconditioned
    z    = workspace.correction
    s    = workspace.svec        # NOT workspace.solution - that IS x
    t    = workspace.t

    bnorm = max(norm(b), eps(T))
    _residual!(hierarchy, r, A, x, b)
    _reset_residual_history!(workspace)
    rnorm = norm(r)
    _push_residual_norm_history!(workspace, rnorm)
    eps_target = _amg_eps(T, atol, rtol, rnorm)
    rel = rnorm / bnorm
    initial_rel = rel

    if rnorm <= eps_target
        workspace.iterations = 0
        workspace.converged = true
        workspace.last_relative_residual = rel
        _update_cycle_factor!(hierarchy, initial_rel, rel, 0, solver)
        return x
    end

    _copy_amg!(hierarchy, rhat, r)          # shadow, held fixed for the solve
    _fill_amg!(hierarchy, p, zero(T))
    _fill_amg!(hierarchy, v, zero(T))
    rho = one(T); alpha = one(T); omega = one(T)
    rhatnorm = rnorm                        # rhat == r here; refreshed at every reset

    best_rnorm = rnorm
    stall = 0
    stall_limit = 20
    k = 0
    broke = false
    restarts = 0
    max_restarts = 2
    true_checked = false
    while k < itmax
        k += 1
        rho_new = dot(rhat, r)
        # Floor scales like `rho_new` itself (||rhat||*||r||), NOT like ||b||: a warm
        # start has ||r|| << ||b||, so an ||b||-scaled floor trips on iteration 1 and
        # returns `x` untouched with `iterations = 0` - a silent no-op solve.
        if !isfinite(rho_new) || abs(rho_new) <= eps(T)*rhatnorm*rnorm
            if restarts < max_restarts && isfinite(rnorm)
                restarts += 1
                _copy_amg!(hierarchy, rhat, r)
                _fill_amg!(hierarchy, p, zero(T))
                _fill_amg!(hierarchy, v, zero(T))
                rho = one(T); alpha = one(T); omega = one(T)
                rhatnorm = rnorm
                k -= 1
                continue
            end
            k -= 1
            break
        end
        beta = (rho_new / rho) * (alpha / omega)
        if !isfinite(beta)
            k -= 1
            break
        end
        _bicg_p!(hierarchy, p, r, v, beta, omega)

        amg_apply_preconditioner!(y, hierarchy, solver, p)
        KernelAbstractions.synchronize(hierarchy.backend)
        _matvec!(hierarchy, v, A, y)

        rhat_v = dot(rhat, v)
        # No eps floor: this scales with ||v||, which nothing held here bounds. A
        # genuinely tiny value blows `alpha` up, which the next guard catches.
        if !isfinite(rhat_v) || iszero(rhat_v)
            k -= 1
            break
        end
        alpha = rho_new / rhat_v
        if !isfinite(alpha)
            k -= 1
            break
        end

        _bicg_s!(hierarchy, s, r, v, alpha)
        snorm = norm(s)
        if !isfinite(snorm)
            break
        end

        # Half-step convergence, and the landing point for both breakdowns below.
        if snorm <= eps_target
            broke = true
        else
            amg_apply_preconditioner!(z, hierarchy, solver, s)
            KernelAbstractions.synchronize(hierarchy.backend)
            _matvec!(hierarchy, t, A, z)
            tt = dot(t, t)
            if !isfinite(tt) || tt <= zero(T)
                broke = true
            else
                omega = dot(t, s) / tt
                if !isfinite(omega) || omega == zero(T)
                    broke = true
                end
            end
        end

        if broke
            _amg_axpy!(hierarchy, x, alpha, y)
            _copy_amg!(hierarchy, r, s)
            rnorm = snorm
            _push_residual_norm_history!(workspace, rnorm)
            rel = rnorm / bnorm
            break
        end

        _bicg_update!(hierarchy, x, r, y, z, s, t, alpha, omega)
        rnorm = norm(r)
        _push_residual_norm_history!(workspace, rnorm)
        rel = rnorm / bnorm
        if !isfinite(rnorm) || !isfinite(rel)
            break
        end
        if rnorm <= eps_target
            _residual!(hierarchy, r, A, x, b)
            rnorm = norm(r)
            rel = rnorm / bnorm
            true_checked = true
            rnorm <= eps_target && break
            if restarts < max_restarts && isfinite(rnorm)
                restarts += 1
                true_checked = false
                _copy_amg!(hierarchy, rhat, r)
                _fill_amg!(hierarchy, p, zero(T))
                _fill_amg!(hierarchy, v, zero(T))
                rho = one(T); alpha = one(T); omega = one(T)
                rhatnorm = rnorm
                best_rnorm = rnorm
                stall = 0
                continue
            end
            break
        end

        if rnorm < best_rnorm * (one(T) - T(1e-4))
            best_rnorm = rnorm
            stall = 0
        else
            stall += 1
            if stall >= stall_limit; break; end
        end
        rho = rho_new
    end

    if !true_checked
        _residual!(hierarchy, r, A, x, b)
        rnorm = norm(r)
        rel = rnorm / bnorm
    end

    workspace.iterations = k
    workspace.converged = rnorm <= eps_target
    workspace.last_relative_residual = rel
    _update_cycle_factor!(hierarchy, initial_rel, rel, k, solver)
    return x
end
