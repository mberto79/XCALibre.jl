@kernel function _amg_coarse_diagonal_solve_kernel!(x, rowptr, colval, nzval, b)
    i = @index(Global)
    T = eltype(x)
    value = zero(T)
    @inbounds for p in rowptr[i]:(rowptr[i + 1] - 1)
        if colval[p] == i
            aii = nzval[p]
            value = abs(aii) > eps(T) ? b[i] / aii : zero(T)
            break
        end
    end
    @inbounds x[i] = value
end

function _is_diagonal_matrix(A)
    _m(A) == _n(A) || return false
    rowptr = _rowptr(A)
    colval = _colval(A)
    @inbounds for i in 1:_m(A)
        row_has_diagonal = false
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            colval[p] == i || return false
            row_has_diagonal = true
        end
        row_has_diagonal || return false
    end
    return true
end

function _coarse_solve_on_device!(hierarchy::AMGHierarchy, level::AMGLevel, b)
    _launch_amg_kernel!(
        hierarchy,
        _amg_coarse_diagonal_solve_kernel!,
        _m(level.A),
        level.x,
        _rowptr(level.A),
        _colval(level.A),
        _nzval(level.A),
        b
    )
    return level.x
end

function _coarse_solve!(coarse_cpu::AMGCPUCoarseLevel, b)
    b === coarse_cpu.rhs || copyto!(coarse_cpu.rhs, b)
    if coarse_cpu.use_qr
        copyto!(coarse_cpu.x, coarse_cpu.qr_factor \ coarse_cpu.rhs)
    else
        ldiv!(coarse_cpu.x, coarse_cpu.lu_factor, coarse_cpu.rhs)
    end
    return coarse_cpu.x
end

function _record_coarse_solve_timing!(hierarchy::AMGHierarchy, rhs_copy_s, cpu_solve_s, x_copy_s)
    hierarchy.coarse_rhs_copy_time_s += rhs_copy_s
    hierarchy.coarse_cpu_solve_time_s += cpu_solve_s
    hierarchy.coarse_x_copy_time_s += x_copy_s
    hierarchy.coarse_solve_calls += 1
    return hierarchy
end

function _coarse_solve!(hierarchy::AMGHierarchy, solver::AMG, level::AMGLevel, b)
    if !(hierarchy.backend isa CPU) && _is_diagonal_matrix(hierarchy.host_levels[end].A)
        return _coarse_solve_on_device!(hierarchy, level, b)
    end
    coarse_cpu = hierarchy.coarse_cpu
    rhs_copy_s = @elapsed _cpu_copyto!(coarse_cpu.rhs, b)
    cpu_solve_s = @elapsed _coarse_solve!(coarse_cpu, coarse_cpu.rhs)
    x_copy_s = @elapsed _device_copyto!(hierarchy.backend, level.x, coarse_cpu.x)
    _record_coarse_solve_timing!(hierarchy, rhs_copy_s, cpu_solve_s, x_copy_s)
    return level.x
end

function _cycle!(hierarchy::AMGHierarchy, cycle::VCycle, solver::AMG, level_index, rhs)
    levels = hierarchy.levels
    level = levels[level_index]
    _fill_amg!(hierarchy, level.x, zero(eltype(level.x)))

    if level_index == length(levels)
        return _coarse_solve!(hierarchy, solver, level, rhs)
    end

    _apply_level_smoother!(hierarchy, solver.smoother, level, rhs, solver.pre_sweeps)
    _residual!(hierarchy, level.rhs, level.A, level.x, rhs)

    coarse_level = levels[level_index + 1]
    _restrict!(hierarchy, coarse_level.rhs, level.R, level.rhs)
    _cycle!(hierarchy, cycle, solver, level_index + 1, coarse_level.rhs)

    _prolongate_add!(hierarchy, level.x, level.P, coarse_level.x, level.tmp)
    _apply_level_smoother!(hierarchy, solver.smoother, level, rhs, solver.post_sweeps)
    return level.x
end

function amg_apply_preconditioner!(z, hierarchy::AMGHierarchy, solver::AMG, r)
    root = hierarchy.levels[1]
    _cycle!(hierarchy, solver.cycle, solver, 1, r)
    _copy_amg!(hierarchy, z, root.x)
    return z
end

function _update_cycle_factor!(hierarchy::AMGHierarchy, initial_rel, final_rel, iterations, solver::AMG)
    if iterations > 0 && initial_rel > 0
        hierarchy.last_cycle_factor = (final_rel / initial_rel)^(1 / iterations)
    else
        hierarchy.last_cycle_factor = 0.0
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
        _add_amg!(hierarchy, x, workspace.correction)
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
