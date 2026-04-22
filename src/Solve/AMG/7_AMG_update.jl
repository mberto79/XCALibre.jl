function _pattern_matches(hierarchy::AMGHierarchy, A)
    hierarchy.nrows == _m(A) || return false
    hierarchy.nnz == length(_nzval(A)) || return false
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

function update!(workspace::AMGWorkspace, A, solver::AMG, config)
    if isnothing(workspace.hierarchy)
        (; hardware) = config
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        workspace.hierarchy = setup_hierarchy(setup_matrix, solver, setup_backend; log_diagnostics=true)
        return workspace
    end

    hierarchy = workspace.hierarchy
    if hierarchy.force_rebuild || !_pattern_matches(hierarchy, A)
        (; hardware) = config
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        workspace.hierarchy = setup_hierarchy(setup_matrix, solver, setup_backend; log_diagnostics=false)
        return workspace
    end

    _sync_finest_matrix!(hierarchy, A)
    if _needs_numeric_refresh(hierarchy, hierarchy.levels[1].A, solver)
        refresh_hierarchy!(hierarchy, solver)
    else
        refresh_finest_level!(hierarchy, solver)
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
        status=workspace.iterations == itmax ? "itmax" : "converged"
    )
    workspace.iterations == itmax && @warn "Maximum number of iterations reached!"
    return residual(phiEqn, component, config)
end
