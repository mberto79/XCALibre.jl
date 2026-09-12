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

"""
    _is_symmetric(A; rtol=1e-9, atol=1e-14) -> Bool

Symmetry test for the `Cg()` gate, measured RELATIVE to the local matrix scale.

"""
function _is_symmetric(A; rtol=1e-9, atol=1e-10)
    rowptr = _rowptr(A); colval = _colval(A); nzval = _nzval(A)
    n = _m(A)
    T = real(eltype(nzval))
    dg = zeros(T, n)
    for i in 1:n
        d = spindex(rowptr, colval, i, i)
        dg[i] = d == 0 ? zero(T) : abs(nzval[d])
    end
    for i in 1:n
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            q = spindex(rowptr, colval, j, i)
            q == 0 && return false
            # `max`, NOT `+`: the relative term ADDS tolerance for a
            # large-scaled matrix but must never REMOVE any. A first attempt used
            # `atol + rtol*scale` with atol = 1e-14, which is far TIGHTER than the
            # original absolute 1e-10 whenever the diagonal is small - and it
            # promptly failed a case that had previously passed, at a LOWER heat
            # flux than the one it was meant to fix.
            tol = max(atol, rtol*max(dg[i], dg[j]))
            abs(nzval[p] - nzval[q]) <= tol || return false
        end
    end
    return true
end

"""
    symmetry_report(A) -> NamedTuple

Worst asymmetry in `A`, absolute and relative to the local diagonal, with the
entry it occurred at and the diagonal magnitudes there.

Exists because `AMG(mode=Cg())` used to fail with a bare "not symmetric" throw,
which says nothing about whether the matrix is badly asymmetric or merely
scaled such that a fixed threshold bites. Those need opposite responses.
"""
function symmetry_report(A)
    rowptr = _rowptr(A); colval = _colval(A); nzval = _nzval(A)
    n = _m(A)
    T = real(eltype(nzval))
    dg = zeros(T, n)
    for i in 1:n
        d = spindex(rowptr, colval, i, i)
        dg[i] = d == 0 ? zero(T) : abs(nzval[d])
    end
    worst_abs = zero(T); worst_rel = zero(T); wi = 0; wj = 0; missing_pairs = 0
    for i in 1:n
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            q = spindex(rowptr, colval, j, i)
            if q == 0
                missing_pairs += 1
                continue
            end
            d = abs(nzval[p] - nzval[q])
            sc = max(dg[i], dg[j])
            r = sc > 0 ? d/sc : (d > 0 ? T(Inf) : zero(T))
            if r > worst_rel
                worst_rel = r; worst_abs = d; wi = i; wj = j
            end
        end
    end
    return (worst_abs = worst_abs, worst_rel = worst_rel, i = wi, j = wj,
            diag_i = wi == 0 ? zero(T) : dg[wi], diag_j = wj == 0 ? zero(T) : dg[wj],
            diag_max = isempty(dg) ? zero(T) : maximum(dg),
            structurally_missing = missing_pairs)
end

# Matches Krylov.jl stopping threshold so swapping Cg()<->AMG keeps tuned tolerances valid
_amg_eps(::Type{T}, atol, rtol, r0norm) where {T} = T(atol) + T(rtol) * r0norm

# scale_correction makes M nonlinear; flexible PR+ beta avoids the FR beta assumption
_amg_cg_flexible(solver::AMG) = solver.scale_correction

function amg_cg_solve!(workspace::AMGWorkspace, hierarchy::AbstractAMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    if !hierarchy.is_symmetric
        r = symmetry_report(A)
        throw(ArgumentError(string(
            "AMG(mode=Cg()) requires a symmetric matrix.
",
            "  worst |A[i,j]-A[j,i]| = ", r.worst_abs, "  at (", r.i, ",", r.j, ")
",
            "  relative to local diagonal = ", r.worst_rel,
            "   (diag there ", r.diag_i, " / ", r.diag_j, ", max diag ", r.diag_max, ")
",
            "  structurally missing transpose entries = ", r.structurally_missing, "

",
            "A relative value near machine precision means the matrix IS symmetric and the
",
            "test scale is wrong. A relative value of order 0.1-1 means it is genuinely
",
            "non-symmetric - use AMG(mode=AMGSolver()) or Bicgstab() instead.")))
    end
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
