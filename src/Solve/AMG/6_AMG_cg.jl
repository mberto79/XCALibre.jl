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
    _is_symmetric(A) || throw(ArgumentError("AMG(mode=:cg) requires a symmetric matrix"))
    T = eltype(x)
    r = workspace.residual
    z = workspace.preconditioned
    p = workspace.search
    q = workspace.q

    _residual!(r, A, x, b)
    bnorm = max(norm(b), eps(T))
    amg_apply_preconditioner!(z, hierarchy, solver, r)
    copyto!(p, z)
    rz = dot(r, z)
    rel = norm(r) / bnorm
    k = 0
    while k < itmax && norm(r) > atol && rel > rtol
        k += 1
        mul!(q, A, p)
        α = rz / dot(p, q)
        @inbounds for i in eachindex(x)
            x[i] += α * p[i]
            r[i] -= α * q[i]
        end
        rel = norm(r) / bnorm
        if norm(r) <= atol || rel <= rtol
            break
        end
        amg_apply_preconditioner!(z, hierarchy, solver, r)
        rz_new = dot(r, z)
        β = rz_new / rz
        @inbounds for i in eachindex(p)
            p[i] = z[i] + β * p[i]
        end
        rz = rz_new
    end
    workspace.iterations = k
    workspace.last_relative_residual = rel
    return x
end
