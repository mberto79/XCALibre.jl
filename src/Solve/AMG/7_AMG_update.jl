function _pattern_matches(hierarchy::AMGHierarchy, A, solver::AMG)
    hierarchy.nrows == _m(A) || return false
    hierarchy.nnz == length(_nzval(A)) || return false
    solver.assume_fixed_pattern && return true
    rowptr, colval = _pattern_signature(A)
    rowptr == hierarchy.rowptr_pattern || return false
    colval == hierarchy.colval_pattern || return false
    return true
end

function _sync_finest_matrix!(hierarchy::AMGHierarchy, A)
    level = hierarchy.levels[1]
    copyto!(_nzval(level.A), _cpu_vector(_nzval(A)))
    return hierarchy
end

function _timing_snapshot(workspace::AMGWorkspace)
    timing = workspace.timing
    return (
        build_time_s=timing.build_time_s,
        build_calls=timing.build_calls,
        refresh_time_s=timing.refresh_time_s,
        refresh_calls=timing.refresh_calls,
        finest_refresh_time_s=timing.finest_refresh_time_s,
        finest_refresh_calls=timing.finest_refresh_calls,
        apply_time_s=timing.apply_time_s,
        apply_calls=timing.apply_calls
    )
end

function _timing_delta(workspace::AMGWorkspace, before)
    timing = workspace.timing
    return _timing_payload(
        build_time_s=timing.build_time_s - before.build_time_s,
        build_calls=timing.build_calls - before.build_calls,
        refresh_time_s=timing.refresh_time_s - before.refresh_time_s,
        refresh_calls=timing.refresh_calls - before.refresh_calls,
        finest_refresh_time_s=timing.finest_refresh_time_s - before.finest_refresh_time_s,
        finest_refresh_calls=timing.finest_refresh_calls - before.finest_refresh_calls,
        apply_time_s=timing.apply_time_s - before.apply_time_s,
        apply_calls=timing.apply_calls - before.apply_calls,
        last_update_action=workspace.timing.last_update_action
    )
end

function update!(workspace::AMGWorkspace, A, solver::AMG, config)
    if isnothing(workspace.hierarchy)
        (; hardware) = config
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        elapsed_s = @elapsed workspace.hierarchy = setup_hierarchy(setup_matrix, solver, setup_backend; log_diagnostics=true)
        _record_build_timing!(workspace, elapsed_s; rebuilt=false)
        return workspace
    end

    hierarchy = workspace.hierarchy
    if hierarchy.force_rebuild || !_pattern_matches(hierarchy, A, solver)
        (; hardware) = config
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        elapsed_s = @elapsed workspace.hierarchy = setup_hierarchy(setup_matrix, solver, setup_backend; log_diagnostics=false)
        _record_build_timing!(workspace, elapsed_s; rebuilt=true)
        return workspace
    end

    _sync_finest_matrix!(hierarchy, A)
    if _needs_numeric_refresh(hierarchy, hierarchy.levels[1].A, solver)
        elapsed_s = @elapsed refresh_hierarchy!(hierarchy, solver)
        _record_refresh_timing!(workspace, elapsed_s)
    else
        elapsed_s = @elapsed refresh_finest_level!(hierarchy, solver)
        _record_finest_refresh_timing!(workspace, elapsed_s)
    end
    return workspace
end

function solve_system!(phiEqn::ModelEquation, setup::SolverSetup{F,I,S1,S2,PT}, result, component, config) where {F,I,S1<:AMG,S2,PT}
    (; itmax, atol, rtol, solver) = setup
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    workspace = phiEqn.solver
    (; hardware) = config
    values = get_values(result, component)
    timing_before = _timing_snapshot(workspace)
    update!(workspace, A, solver, config)
    apply_smoother!(setup.smoother, values, A, b, hardware)
    x = workspace.solution
    copyto!(x, values)

    if solver.mode == :solver
        amg_solve!(workspace, workspace.hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    else
        amg_cg_solve!(workspace, workspace.hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    end

    if typeof(phiEqn.model.terms[1].type) <: Time{CrankNicolson}
        xcal_foreach(x, config) do i
            x[i] = 2 * x[i] - values[i]
        end
    end

    copyto!(values, x)
    _record_linear_solve!(
        phiEqn,
        setup,
        component,
        workspace.iterations,
        itmax,
        workspace.residual_history;
        status=_amg_status(workspace, itmax),
        hit_itmax=_amg_hit_itmax(workspace, itmax),
        timing=_timing_delta(workspace, timing_before)
    )
    _amg_hit_itmax(workspace, itmax) && @warn "Maximum number of iterations reached!"
    return residual(phiEqn, component, config)
end
