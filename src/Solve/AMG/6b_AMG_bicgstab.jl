const BICG_REASON = Ref(:none)
# =============================================================================
#  AMG-preconditioned BiCGStab
# =============================================================================
#
#  WHY THIS EXISTS
#
#  `AMG` previously offered two modes, and neither suits a NON-SYMMETRIC matrix:
#
#    Cg()          Krylov accelerated, but requires symmetry. Used on the LH2
#                  pipe under `pressure_form = :mass` for months and it did
#                  converge - but that matrix is 9.5% asymmetric relative to its
#                  diagonal (measured), so it was never valid. It passed the old
#                  ABSOLUTE 1e-10 symmetry test only because the matrix itself is
#                  small (diagonal ~1e-7), and threw as soon as rising void grew
#                  `psi` enough.
#
#    AMGSolver()   valid for any matrix, but a plain fixed-point iteration: apply
#                  the V-cycle, add the correction, recompute the residual. NO
#                  Krylov acceleration, so it needs far more cycles - measured
#                  ~5x the cost of Cg-mode on the same problem.
#
#  The asymmetry is `Divergence(pconv, p_rgh)`, the implicit pressure convection.
#  It is upwinded and therefore non-symmetric by construction, and it scales as
#  `psi*|U.Sf|` - so it GROWS with void fraction and cannot be tuned away.
#
#  BiCGStab is the standard answer: a short-recurrence Krylov method that makes no
#  symmetry assumption, driven by the SAME V-cycle preconditioner. It recovers the
#  acceleration `AMGSolver()` lacks while staying valid for the operator that is
#  actually assembled.
#
#  COST PER ITERATION. Two preconditioner applications and two matvecs, against
#  one of each for CG - so roughly 2x a CG iteration, but the iteration COUNT does
#  not blow up the way an unaccelerated fixed-point iteration does. The comparison
#  that matters is against `AMGSolver()`, not against a CG that was never entitled
#  to run on this matrix.
# =============================================================================

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

"""
    amg_bicgstab_solve!(workspace, hierarchy, solver, A, b, x; itmax, atol, rtol)

Preconditioned BiCGStab with the AMG V-cycle as `M^-1`. No symmetry requirement.

The two solution updates (`x += alpha*y` then `x += omega*z`) and the residual
update are FUSED into a single kernel launch, so an iteration costs 2
preconditioner applies, 2 matvecs and 4 vector kernels rather than the 7 a
literal transcription of the algorithm needs.

### Breakdown handling

BiCGStab can break down two ways, and both are detected rather than left to
produce silent nonsense:

  * `rho -> 0` - the shadow residual has gone orthogonal to the residual.
  * `omega -> 0` - `t` has collapsed, so the stabilising minimisation is
    undefined.

In both cases the HALF-STEP iterate (`x += alpha*y`, `r = s`) is kept, because it
is a genuine improvement on the incoming `x`, and the solve reports the iteration
count it actually reached. Restarting with a fresh shadow is the textbook remedy
but changes the Krylov space mid-solve, which interacts badly with a nonlinear
(scale-corrected) preconditioner.

There is also a stall guard matching [`amg_cg_solve!`](@ref): 20 iterations
without a 1e-4 relative improvement ends the solve. A stalled solve that reports
success has cost this project weeks before - see `record_linear_solve!`.
"""
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

    best_rnorm = rnorm
    stall = 0
    stall_limit = 20
    k = 0
    broke = false
    restarts = 0
    max_restarts = 2
    while k < itmax
        k += 1
        rho_new = dot(rhat, r)
        if !isfinite(rho_new) || abs(rho_new) <= eps(T)*bnorm
            # RHO BREAKDOWN: the fixed shadow has gone orthogonal to the current
            # residual. Near convergence this is the normal way BiCGStab ends, and
            # giving up here left a solve at 6.54e-9 against a 6.32e-9 target -
            # seven good iterations thrown away for want of an eighth.
            #
            # Restarting with `rhat = r` rebuilds a usable shadow. It DOES change
            # the Krylov space mid-solve, which is why it is capped: two restarts,
            # then accept the iterate. Unlimited restarts on a genuinely broken
            # system would spin to itmax achieving nothing.
            if restarts < max_restarts && isfinite(rnorm)
                restarts += 1
                _copy_amg!(hierarchy, rhat, r)
                _fill_amg!(hierarchy, p, zero(T))
                _fill_amg!(hierarchy, v, zero(T))
                rho = one(T); alpha = one(T); omega = one(T)
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
        if !isfinite(rhat_v) || abs(rhat_v) <= eps(T)*bnorm
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
        if rnorm <= eps_target; reason = :converged; break; end

        if rnorm < best_rnorm * (one(T) - T(1e-4))
            best_rnorm = rnorm
            stall = 0
        else
            stall += 1
            if stall >= stall_limit; reason = :stall; break; end
        end
        rho = rho_new
    end

    workspace.iterations = k
    workspace.converged = rnorm <= eps_target
    workspace.last_relative_residual = rel
    _update_cycle_factor!(hierarchy, initial_rel, rel, k, solver)
    return x
end
