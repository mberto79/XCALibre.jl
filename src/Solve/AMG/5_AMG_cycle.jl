function _coarse_solve!(level::AMGLevel, b)
    solver = isnothing(level.coarse_solver) ? factorize(_csr_to_csc(level.A)) : level.coarse_solver
    xc = solver \ _cpu_vector(b)
    copyto!(level.x, xc)
    return level.x
end

function _vcycle!(hierarchy::AMGHierarchy, solver::AMG, level_index, rhs)
    levels = hierarchy.levels
    level = levels[level_index]
    fill!(level.x, zero(eltype(level.x)))

    if level_index == length(levels)
        return _coarse_solve!(level, rhs)
    end

    _apply_level_smoother!(solver.smoother, level, rhs, solver.presweeps)
    _residual!(level.rhs, level.A, level.x, rhs)

    coarse_level = levels[level_index + 1]
    _restrict!(coarse_level.rhs, level.R, level.rhs)
    fill!(coarse_level.x, zero(eltype(coarse_level.x)))
    _vcycle!(hierarchy, solver, level_index + 1, coarse_level.rhs)

    _prolongate_add!(level.x, level.P, coarse_level.x, level.tmp)
    _apply_level_smoother!(solver.smoother, level, rhs, solver.postsweeps)
    return level.x
end

function amg_apply_preconditioner!(z, hierarchy::AMGHierarchy, solver::AMG, r)
    root = hierarchy.levels[1]
    fill!(root.x, zero(eltype(root.x)))
    _vcycle!(hierarchy, solver, 1, r)
    copyto!(z, root.x)
    return z
end

function _amg_cpu_solve!(workspace::AMGWorkspace, hierarchy::AMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    Acpu = hierarchy.levels[1].A
    bcpu = Array(b)
    xcpu = Array(x)
    cpu_workspace = AMGWorkspace(
        hierarchy,
        similar(bcpu),
        similar(bcpu),
        similar(bcpu),
        similar(bcpu),
        similar(bcpu),
        0,
        zero(eltype(bcpu))
    )

    if solver.mode == :solver
        amg_solve!(cpu_workspace, hierarchy, solver, Acpu, bcpu, xcpu; itmax=itmax, atol=atol, rtol=rtol)
    else
        amg_cg_solve!(cpu_workspace, hierarchy, solver, Acpu, bcpu, xcpu; itmax=itmax, atol=atol, rtol=rtol)
    end

    copyto!(x, xcpu)
    workspace.iterations = cpu_workspace.iterations
    workspace.last_relative_residual = cpu_workspace.last_relative_residual
    return x
end

function amg_solve!(workspace::AMGWorkspace, hierarchy::AMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    _amg_needs_cpu_apply(A) && return _amg_cpu_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    hierarchy.backend isa CPU || return _amg_cpu_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    T = eltype(x)
    _residual!(workspace.residual, A, x, b)
    bnorm = max(norm(b), eps(T))
    rel = norm(workspace.residual) / bnorm
    it = 0
    while it < itmax && norm(workspace.residual) > atol && rel > rtol
        it += 1
        amg_apply_preconditioner!(workspace.correction, hierarchy, solver, workspace.residual)
        @inbounds for i in eachindex(x)
            x[i] += workspace.correction[i]
        end
        _residual!(workspace.residual, A, x, b)
        rel = norm(workspace.residual) / bnorm
    end
    workspace.iterations = it
    workspace.last_relative_residual = rel
    return x
end
