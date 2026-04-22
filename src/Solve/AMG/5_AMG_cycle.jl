function _coarse_solve!(level::AMGLevel, b)
    solver = isnothing(level.coarse_solver) ? _build_coarse_solver(level.A) : level.coarse_solver
    bcpu = _cpu_vector(b)
    try
        ldiv!(level.x, solver, bcpu)
    catch
        xc = solver \ bcpu
        copyto!(level.x, xc)
    end
    return level.x
end

_cycle_repetitions(solver::AMG) = solver.cycle == :W ? 2 : 1

function _cycle!(hierarchy::AMGHierarchy, solver::AMG, level_index, rhs)
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
    for _ in 1:_cycle_repetitions(solver)
        _cycle!(hierarchy, solver, level_index + 1, coarse_level.rhs)
    end

    _prolongate_add!(level.x, level.P, coarse_level.x, level.tmp)
    _apply_level_smoother!(solver.smoother, level, rhs, solver.postsweeps)
    return level.x
end

function amg_apply_preconditioner!(z, hierarchy::AMGHierarchy, solver::AMG, r)
    root = hierarchy.levels[1]
    fill!(root.x, zero(eltype(root.x)))
    _cycle!(hierarchy, solver, 1, r)
    copyto!(z, root.x)
    return z
end

function _update_cycle_factor!(hierarchy::AMGHierarchy, initial_rel, final_rel, iterations, solver::AMG)
    if iterations > 0 && initial_rel > 0
        hierarchy.last_cycle_factor = (final_rel / initial_rel)^(1 / iterations)
        hierarchy.force_rebuild = hierarchy.last_cycle_factor > solver.adaptive_rebuild_factor
    else
        hierarchy.last_cycle_factor = 0.0
        hierarchy.force_rebuild = false
    end
    return hierarchy.last_cycle_factor
end

function _reset_residual_history!(workspace::AMGWorkspace)
    empty!(workspace.residual_history)
    return workspace
end

function _push_residual_history!(workspace::AMGWorkspace, residual)
    push!(workspace.residual_history, float(norm(residual)))
    return workspace
end

function _ensure_cpu_workspace!(hierarchy::AMGHierarchy, b, x)
    needs_init = isnothing(hierarchy.cpu_workspace) || isnothing(hierarchy.cpu_rhs) || isnothing(hierarchy.cpu_x) ||
        length(hierarchy.cpu_rhs) != length(b) || length(hierarchy.cpu_x) != length(x)
    if needs_init
        bcpu = Array(b)
        xcpu = Array(x)
        hierarchy.cpu_rhs = similar(bcpu)
        hierarchy.cpu_x = similar(xcpu)
        hierarchy.cpu_workspace = AMGWorkspace(
            hierarchy,
            similar(bcpu),
            similar(bcpu),
            similar(bcpu),
            similar(bcpu),
            similar(bcpu),
            similar(bcpu),
            0,
            zero(eltype(bcpu)),
            Float64[]
        )
    end
    return hierarchy.cpu_workspace, hierarchy.cpu_rhs, hierarchy.cpu_x
end

function _amg_cpu_solve!(workspace::AMGWorkspace, hierarchy::AMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    Acpu = hierarchy.levels[1].A
    cpu_workspace, bcpu, xcpu = _ensure_cpu_workspace!(hierarchy, b, x)
    copyto!(bcpu, Array(b))
    copyto!(xcpu, Array(x))

    if solver.mode == :solver
        amg_solve!(cpu_workspace, hierarchy, solver, Acpu, bcpu, xcpu; itmax=itmax, atol=atol, rtol=rtol)
    else
        amg_cg_solve!(cpu_workspace, hierarchy, solver, Acpu, bcpu, xcpu; itmax=itmax, atol=atol, rtol=rtol)
    end

    copyto!(x, xcpu)
    workspace.iterations = cpu_workspace.iterations
    workspace.last_relative_residual = cpu_workspace.last_relative_residual
    empty!(workspace.residual_history)
    append!(workspace.residual_history, cpu_workspace.residual_history)
    return x
end

function amg_solve!(workspace::AMGWorkspace, hierarchy::AMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    _amg_needs_cpu_apply(A) && return _amg_cpu_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    hierarchy.backend isa CPU || return _amg_cpu_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    T = eltype(x)
    _residual!(workspace.residual, A, x, b)
    _reset_residual_history!(workspace)
    _push_residual_history!(workspace, workspace.residual)
    bnorm = max(norm(b), eps(T))
    rel = norm(workspace.residual) / bnorm
    initial_rel = rel
    it = 0
    while it < itmax && norm(workspace.residual) > atol && rel > rtol
        it += 1
        amg_apply_preconditioner!(workspace.correction, hierarchy, solver, workspace.residual)
        @inbounds for i in eachindex(x)
            x[i] += workspace.correction[i]
        end
        _residual!(workspace.residual, A, x, b)
        _push_residual_history!(workspace, workspace.residual)
        rel = norm(workspace.residual) / bnorm
    end
    workspace.iterations = it
    workspace.last_relative_residual = rel
    _update_cycle_factor!(hierarchy, initial_rel, rel, it, solver)
    return x
end
