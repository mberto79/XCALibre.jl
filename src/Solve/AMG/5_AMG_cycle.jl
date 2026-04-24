function _coarse_solve!(coarse_cpu::AMGCPUCoarseLevel, b)
    copyto!(coarse_cpu.rhs, b)
    if coarse_cpu.use_qr
        ldiv!(coarse_cpu.x, coarse_cpu.qr_factor, coarse_cpu.rhs)
    else
        ldiv!(coarse_cpu.x, coarse_cpu.lu_factor, coarse_cpu.rhs)
    end
    return coarse_cpu.x
end

function _coarse_solve!(hierarchy::AMGHierarchy, level::AMGLevel, b)
    coarse_cpu = hierarchy.coarse_cpu
    _cpu_copyto!(coarse_cpu.rhs, b)
    _coarse_solve!(coarse_cpu, coarse_cpu.rhs)
    _device_copyto!(hierarchy.backend, level.x, coarse_cpu.x)
    return level.x
end

function _cycle!(hierarchy::AMGHierarchy, solver::AMG, level_index, rhs)
    levels = hierarchy.levels
    level = levels[level_index]
    _fill_amg!(hierarchy, level.x, zero(eltype(level.x)))

    if level_index == length(levels)
        return _coarse_solve!(hierarchy, level, rhs)
    end

    _apply_level_smoother!(hierarchy, solver.smoother, level, rhs, solver.presweeps)
    _residual!(hierarchy, level.rhs, level.A, level.x, rhs)

    coarse_level = levels[level_index + 1]
    _restrict!(hierarchy, coarse_level.rhs, level.R, level.rhs)
    _fill_amg!(hierarchy, coarse_level.x, zero(eltype(coarse_level.x)))
    _cycle!(hierarchy, solver, level_index + 1, coarse_level.rhs)

    _prolongate_add!(hierarchy, level.x, level.P, coarse_level.x, level.tmp)
    _apply_level_smoother!(hierarchy, solver.smoother, level, rhs, solver.postsweeps)
    return level.x
end

function amg_apply_preconditioner!(z, hierarchy::AMGHierarchy, solver::AMG, r)
    root = hierarchy.levels[1]
    _fill_amg!(hierarchy, root.x, zero(eltype(root.x)))
    _cycle!(hierarchy, solver, 1, r)
    _copy_amg!(hierarchy, z, root.x)
    return z
end

function _update_cycle_factor!(hierarchy::AMGHierarchy, initial_rel, final_rel, iterations, solver::AMG)
    if iterations > 0 && initial_rel > 0
        hierarchy.last_cycle_factor = (final_rel / initial_rel)^(1 / iterations)
        if solver.mode == :cg
            hierarchy.force_rebuild = false
            return hierarchy.last_cycle_factor
        end
        hierarchy.force_rebuild = hierarchy.reuse_steps > 0 &&
            hierarchy.last_cycle_factor > solver.adaptive_rebuild_factor
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

function _push_residual_norm_history!(workspace::AMGWorkspace, residual_norm)
    push!(workspace.residual_history, float(residual_norm))
    return workspace
end

function _record_build_timing!(workspace::AMGWorkspace, elapsed_s; rebuilt=false)
    workspace.timing.build_time_s += elapsed_s
    workspace.timing.build_calls += 1
    workspace.timing.last_update_action = rebuilt ? :rebuild : :build
    return workspace
end

function _record_refresh_timing!(workspace::AMGWorkspace, elapsed_s)
    workspace.timing.refresh_time_s += elapsed_s
    workspace.timing.refresh_calls += 1
    workspace.timing.last_update_action = :refresh
    return workspace
end

function _record_finest_refresh_timing!(workspace::AMGWorkspace, elapsed_s)
    workspace.timing.finest_refresh_time_s += elapsed_s
    workspace.timing.finest_refresh_calls += 1
    workspace.timing.last_update_action = :finest_refresh
    return workspace
end

function _record_apply_timing!(workspace::AMGWorkspace, elapsed_s)
    workspace.timing.apply_time_s += elapsed_s
    workspace.timing.apply_calls += 1
    return workspace
end

function amg_solve!(workspace::AMGWorkspace, hierarchy::AMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    T = eltype(x)
    _residual!(hierarchy, workspace.residual, A, x, b)
    _reset_residual_history!(workspace)
    bnorm = max(norm(b), eps(T))
    rnorm = norm(workspace.residual)
    _push_residual_norm_history!(workspace, rnorm)
    rel = rnorm / bnorm
    initial_rel = rel
    it = 0
    while it < itmax && rnorm > atol && rel > rtol
        it += 1
        elapsed_s = @elapsed begin
            amg_apply_preconditioner!(workspace.correction, hierarchy, solver, workspace.residual)
            KernelAbstractions.synchronize(hierarchy.backend)
        end
        _record_apply_timing!(workspace, elapsed_s)
        _launch_amg_kernel!(hierarchy, _amg_add_kernel!, length(x), x, workspace.correction)
        _residual!(hierarchy, workspace.residual, A, x, b)
        rnorm = norm(workspace.residual)
        _push_residual_norm_history!(workspace, rnorm)
        rel = rnorm / bnorm
    end
    workspace.iterations = it
    workspace.last_relative_residual = rel
    _update_cycle_factor!(hierarchy, initial_rel, rel, it, solver)
    return x
end
