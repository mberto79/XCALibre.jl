function _pattern_matches(hierarchy::AMGHierarchy, A)
    hierarchy.nrows == _m(A) || return false
    hierarchy.nnz == length(_nzval(A)) || return false
    rp = _rowptr(A)
    cv = _colval(A)
    # fast path: same underlying array objects as at setup
    rp === hierarchy.rowptr_ref[] && cv === hierarchy.colval_ref[] && return true
    rowptr = _cpu_vector(rp)
    colval = _cpu_vector(cv)
    length(rowptr) == length(hierarchy.rowptr_pattern) || return false
    length(colval) == length(hierarchy.colval_pattern) || return false
    # medium path: hash of the pattern
    h = hash(colval, hash(rowptr))
    if h == hierarchy.pattern_hash
        hierarchy.rowptr_ref[] = rp
        hierarchy.colval_ref[] = cv
        return true
    end
    # slow path: full equality
    @inbounds for i in eachindex(rowptr)
        Int(rowptr[i]) == hierarchy.rowptr_pattern[i] || return false
    end
    @inbounds for i in eachindex(colval)
        Int(colval[i]) == hierarchy.colval_pattern[i] || return false
    end
    hierarchy.rowptr_ref[] = rp
    hierarchy.colval_ref[] = cv
    hierarchy.pattern_hash = h
    return true
end

function _sync_finest_matrix!(hierarchy::AMGHierarchy, A)
    level = hierarchy.host_levels[1]
    _cpu_copyto!(_nzval(level.A), _nzval(A))
    return hierarchy
end

# Device-resident finest-level refresh: copy nzval device-to-device, recompute diag on device,
# reuse lambda_max from build. Avoids the full nzval D->H/H->D round-trip and host power iteration.
function _refresh_finest_level_device!(hierarchy::AMGHierarchy, A)
    backend = hierarchy.backend
    dev = hierarchy.levels[1]
    _device_copyto!(backend, _nzval(dev.A), _nzval(A))
    _launch_amg_kernel!(
        hierarchy,
        _amg_extract_diagonal_kernel!,
        length(dev.diagonal),
        dev.diagonal,
        dev.inv_diagonal,
        _nzval(dev.A),
        dev.diagonal_index
    )
    return hierarchy
end

function _sync_workspace_hierarchy!(workspace::AMGWorkspace, hierarchy)
    workspace.hierarchy = hierarchy
    return workspace
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
    (; hardware) = config
    hierarchy = workspace.hierarchy
    hierarchy.backend = hardware.backend
    hierarchy.workgroup = _amg_workgroup(hardware.backend, hardware.workgroup, max(_m(A), 1))

    if isempty(hierarchy.host_levels)
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        elapsed_s = @elapsed begin
            workspace.hierarchy = setup_hierarchy(setup_matrix, solver, hardware.backend, hardware.workgroup; log_diagnostics=true)
        end
        _record_build_timing!(workspace, elapsed_s; rebuilt=false)
        return workspace
    end

    if !_pattern_matches(hierarchy, A)
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        elapsed_s = @elapsed begin
            workspace.hierarchy = setup_hierarchy(setup_matrix, solver, hardware.backend, hardware.workgroup; log_diagnostics=false)
        end
        _record_build_timing!(workspace, elapsed_s; rebuilt=true)
        return workspace
    end

    numeric_updates = workspace.timing.refresh_calls + workspace.timing.finest_refresh_calls
    refresh_coarse = (numeric_updates + 1) % solver.coarse_refresh_interval == 0
    if !refresh_coarse
        elapsed_s = @elapsed begin
            if hierarchy.backend isa CPU
                _sync_finest_matrix!(hierarchy, A)
                refresh_finest_level!(hierarchy, solver)
            else
                _refresh_finest_level_device!(hierarchy, A)
            end
        end
        _record_finest_refresh_timing!(workspace, elapsed_s)
        return workspace
    end

    elapsed_s = @elapsed begin
        _sync_finest_matrix!(hierarchy, A)
        refresh_hierarchy!(hierarchy, solver)
        _sync_device_levels_numeric!(hierarchy)
    end
    _record_refresh_timing!(workspace, elapsed_s)
    return workspace
end

function solve_system!(phiEqn::ModelEquation, setup::SolverSetup{F,I,S1,S2,PT}, result, component, config) where {F,I,S1<:AMG,S2,PT}
    (; itmax, atol, rtol, solver) = setup
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    workspace = phiEqn.solver
    values = get_values(result, component)
    timing_before = _timing_snapshot(workspace)
    update!(workspace, A, solver, config)
    apply_smoother!(setup.smoother, values, A, b, config.hardware)
    x = workspace.solution
    copyto!(x, values)
    _record_pressure_matrix_capture!(phiEqn, setup, component, A, b, x)

    fine_A = workspace.hierarchy.levels[1].A
    _amg_solve_mode!(workspace, workspace.hierarchy, solver, solver.mode, fine_A, b, x; itmax=itmax, atol=atol, rtol=rtol)

    if typeof(phiEqn.model.terms[1].type) <: Time{CrankNicolson}
        xcal_foreach(x, config) do i
            x[i] = 2 * x[i] - values[i]
        end
    end

    copyto!(values, x)
    final_abs = isempty(workspace.residual_history) ? Inf : workspace.residual_history[end]
    converged = isfinite(workspace.last_relative_residual) &&
        (workspace.last_relative_residual <= rtol || final_abs <= atol)
    status = converged ? "converged" : workspace.iterations == itmax ? "itmax" : "breakdown"
    _record_linear_solve!(
        phiEqn,
        setup,
        component,
        workspace.iterations,
        itmax,
        workspace.residual_history;
        status=status,
        timing=_timing_delta(workspace, timing_before)
    )
    workspace.iterations == itmax && @warn "Maximum number of iterations reached!"
    (!converged && workspace.iterations != itmax) && @warn "AMG solver stopped before convergence" status=status relative_residual=workspace.last_relative_residual
    return residual(phiEqn, component, config)
end

function _amg_solve_mode!(workspace, hierarchy, solver::AMG, ::AMGSolver, A, b, x; itmax, atol, rtol)
    return amg_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
end

function _amg_solve_mode!(workspace, hierarchy, solver::AMG, ::Cg, A, b, x; itmax, atol, rtol)
    return amg_cg_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
end
