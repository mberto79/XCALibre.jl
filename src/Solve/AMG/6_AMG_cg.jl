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
    hierarchy.is_symmetric || throw(ArgumentError("AMG(mode=:cg) requires a symmetric matrix"))
    T = eltype(x)
    r = workspace.residual
    z = workspace.preconditioned
    p = workspace.search
    q = workspace.q

    _residual!(hierarchy, r, A, x, b)
    _reset_residual_history!(workspace)
    bnorm = max(norm(b), eps(T))
    rnorm = norm(r)
    _push_residual_norm_history!(workspace, rnorm)
    rel = rnorm / bnorm
    initial_rel = rel
    if rnorm <= atol || rel <= rtol
        workspace.iterations = 0
        workspace.last_relative_residual = rel
        _update_cycle_factor!(hierarchy, initial_rel, rel, 0, solver)
        return x
    end

    elapsed_s = @elapsed begin
        amg_apply_preconditioner!(z, hierarchy, solver, r)
        KernelAbstractions.synchronize(hierarchy.backend)
    end
    _record_apply_timing!(workspace, elapsed_s)
    _copy_amg!(hierarchy, p, z)
    rz = dot(r, z)
    k = 0
    while k < itmax && rnorm > atol && rel > rtol
        k += 1
        _matvec!(hierarchy, q, A, p)
        α = rz / dot(p, q)
        _launch_amg_kernel!(hierarchy, _amg_cg_step_kernel!, length(x), x, r, p, q, α)
        rnorm = norm(r)
        _push_residual_norm_history!(workspace, rnorm)
        rel = rnorm / bnorm
        if rnorm <= atol || rel <= rtol
            break
        end
        elapsed_s = @elapsed begin
            amg_apply_preconditioner!(z, hierarchy, solver, r)
            KernelAbstractions.synchronize(hierarchy.backend)
        end
        _record_apply_timing!(workspace, elapsed_s)
        rz_new = dot(r, z)
        β = rz_new / rz
        _launch_amg_kernel!(hierarchy, _amg_xpay_kernel!, length(p), p, z, β)
        rz = rz_new
    end
    workspace.iterations = k
    workspace.last_relative_residual = rel
    _update_cycle_factor!(hierarchy, initial_rel, rel, k, solver)
    return x
end
