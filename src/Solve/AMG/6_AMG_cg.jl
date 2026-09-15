@kernel function _amg_axpy_kernel!(y, alpha, x)
    i = @index(Global)
    @inbounds y[i] += alpha * x[i]
end

@kernel function _amg_xpay_kernel!(y, x, beta)
    i = @index(Global)
    @inbounds y[i] = x[i] + beta * y[i]
end

@kernel function _amg_cg_step_kernel!(x, r, p, q, alpha)
    i = @index(Global)
    @inbounds begin
        x[i] += alpha * p[i]
        r[i] -= alpha * q[i]
    end
end

function _xpay_amg!(hierarchy::AbstractAMGHierarchy, y, x, beta)
    _launch_amg_kernel!(hierarchy, _amg_xpay_kernel!, length(y), y, x, beta)
    return y
end

function _cg_step_amg!(hierarchy::AbstractAMGHierarchy, x, r, p, q, alpha)
    _launch_amg_kernel!(hierarchy, _amg_cg_step_kernel!, length(x), x, r, p, q, alpha)
    return x
end

function _is_symmetric(A; atol=1e-10)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    for i in 1:_m(A)
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            q = spindex(rowptr, colval, j, i)
            q == 0 && return false
            abs(nzval[p] - nzval[q]) <= atol || return false
        end
    end
    return true
end

# Matches Krylov.jl stopping threshold so swapping Cg()<->AMG keeps tuned tolerances valid
_amg_eps(::Type{T}, atol, rtol, r0norm) where {T} = T(atol) + T(rtol) * r0norm

# scale_correction makes M nonlinear; flexible PR+ beta avoids the FR beta assumption
_amg_cg_flexible(solver::AMG) = solver.scale_correction

function amg_cg_solve!(workspace::AMGWorkspace, hierarchy::AbstractAMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    hierarchy.is_symmetric || throw(ArgumentError("AMG(mode=Cg()) requires a symmetric matrix"))
    T = eltype(x)
    r = workspace.residual
    z = workspace.preconditioned
    p = workspace.search
    q = workspace.q
    flex = _amg_cg_flexible(solver)
    r_prev = workspace.correction

    bnorm = max(norm(b), eps(T))

    _residual!(hierarchy, r, A, x, b)
    _reset_residual_history!(workspace)
    rnorm = norm(r)
    _push_residual_norm_history!(workspace, rnorm)
    ε = _amg_eps(T, atol, rtol, rnorm)
    rel = rnorm / bnorm
    initial_rel = rel
    if rnorm <= ε
        workspace.iterations = 0
        workspace.converged = true
        workspace.last_relative_residual = rel
        _update_cycle_factor!(hierarchy, initial_rel, rel, 0, solver)
        return x
    end

    amg_apply_preconditioner!(z, hierarchy, solver, r)
    KernelAbstractions.synchronize(hierarchy.backend)
    _copy_amg!(hierarchy, p, z)
    rz = dot(r, z)
    # No eps(T) floor: pq/rz ~ ||r||^2, so eps floor false-trips for small ||b||
    if !isfinite(rz) || rz <= zero(T)
        workspace.iterations = 0
        workspace.converged = false
        workspace.last_relative_residual = rel
        _update_cycle_factor!(hierarchy, initial_rel, rel, 0, solver)
        return x
    end
    best_rnorm = rnorm
    stall = 0
    stall_limit = 20
    k = 0
    while k < itmax
        k += 1
        _matvec!(hierarchy, q, A, p)
        pq = dot(p, q)
        if !isfinite(pq) || pq <= zero(T)
            k -= 1
            break
        end
        α = rz / pq
        if !isfinite(α)
            k -= 1
            break
        end
        flex && _copy_amg!(hierarchy, r_prev, r)
        _cg_step_amg!(hierarchy, x, r, p, q, α)
        rnorm = norm(r)
        _push_residual_norm_history!(workspace, rnorm)
        rel = rnorm / bnorm
        if !isfinite(rnorm) || !isfinite(rel)
            break
        end
        rnorm <= ε && break
        if rnorm < best_rnorm * (one(T) - T(1e-4))
            best_rnorm = rnorm
            stall = 0
        else
            stall += 1
            stall >= stall_limit && break
        end
        amg_apply_preconditioner!(z, hierarchy, solver, r)
        KernelAbstractions.synchronize(hierarchy.backend)
        rz_new = dot(r, z)
        if !isfinite(rz_new) || rz_new <= zero(T)
            break
        end
        β = flex ? max(zero(T), (rz_new - dot(z, r_prev)) / rz) : rz_new / rz
        if !isfinite(β)
            break
        end
        _xpay_amg!(hierarchy, p, z, β)
        rz = rz_new
    end
    workspace.iterations = k
    workspace.converged = rnorm <= ε
    workspace.last_relative_residual = rel
    _update_cycle_factor!(hierarchy, initial_rel, rel, k, solver)
    return x
end
