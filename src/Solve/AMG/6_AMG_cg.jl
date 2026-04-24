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

function amg_cg_solve!(workspace::AMGWorkspace, hierarchy::AMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    _amg_needs_cpu_apply(A) && return _amg_cpu_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    hierarchy.backend isa CPU || return _amg_cpu_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    hierarchy.is_symmetric || throw(ArgumentError("AMG(mode=:cg) requires a symmetric matrix"))
    T = eltype(x)
    r = workspace.residual
    z = workspace.preconditioned
    p = workspace.search
    q = workspace.q

    _residual!(r, A, x, b)
    _reset_residual_history!(workspace)
    rnorm = norm(r)
    _push_residual_history!(workspace, rnorm)
    bnorm = max(norm(b), eps(T))
    rel = rnorm / bnorm
    initial_rel = rel
    if _amg_converged(rnorm, rel, atol, rtol)
        _set_amg_result!(workspace, 0, rnorm, rel, atol, rtol)
        _update_cycle_factor!(hierarchy, initial_rel, rel, 0, solver)
        return x
    end
    start_ns = time_ns()
    amg_apply_preconditioner!(z, hierarchy, solver, r)
    elapsed_s = _elapsed_seconds(start_ns)
    _record_apply_timing!(workspace, elapsed_s)
    copyto!(p, z)
    rz = dot(r, z)
    k = 0
    while k < itmax && !_amg_converged(rnorm, rel, atol, rtol)
        k += 1
        mul!(q, A, p)
        α = rz / dot(p, q)
        @inbounds for i in eachindex(x)
            x[i] += α * p[i]
            r[i] -= α * q[i]
        end
        rnorm = norm(r)
        _push_residual_history!(workspace, rnorm)
        rel = rnorm / bnorm
        if _amg_converged(rnorm, rel, atol, rtol)
            break
        end
        start_ns = time_ns()
        amg_apply_preconditioner!(z, hierarchy, solver, r)
        elapsed_s = _elapsed_seconds(start_ns)
        _record_apply_timing!(workspace, elapsed_s)
        rz_new = dot(r, z)
        β = rz_new / rz
        @inbounds for i in eachindex(p)
            p[i] = z[i] + β * p[i]
        end
        rz = rz_new
    end
    _set_amg_result!(workspace, k, rnorm, rel, atol, rtol)
    _update_cycle_factor!(hierarchy, initial_rel, rel, k, solver)
    return x
end
