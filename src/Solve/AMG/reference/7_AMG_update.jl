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

function update!(workspace::AMGWorkspace, A, solver::AMG, config)
    (; hardware) = config
    hierarchy = workspace.hierarchy
    hierarchy.backend = hardware.backend
    hierarchy.workgroup = _amg_workgroup(hardware.backend, hardware.workgroup, max(_m(A), 1))

    # Greenfield GPU pipeline owns its own build/refresh (no materialized hierarchy): branch before
    # the reference setup so the VRAM-saving matrix-free path is not double-built.
    if _use_greenfield_amg(solver, hardware.backend)
        return _greenfield_update!(workspace, A, solver, hardware)
    end

    if isempty(hierarchy.host_levels)
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        workspace.hierarchy = setup_hierarchy(setup_matrix, solver, hardware.backend, hardware.workgroup; log_diagnostics=true)
        return workspace
    end

    if !_pattern_matches(hierarchy, A)
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        workspace.hierarchy = setup_hierarchy(setup_matrix, solver, hardware.backend, hardware.workgroup; log_diagnostics=false)
        return workspace
    end

    numeric_updates = workspace.refresh_count
    refresh_coarse = (numeric_updates + 1) % solver.coarse_refresh_interval == 0
    if !refresh_coarse
        if hierarchy.backend isa CPU
            _sync_finest_matrix!(hierarchy, A)
            refresh_finest_level!(hierarchy, solver)
            _amg_mixed_precision(hierarchy) && _sync_storage_levels!(hierarchy)
        else
            _refresh_finest_level_device!(hierarchy, A)
        end
        workspace.refresh_count += 1
        return workspace
    end

    if hierarchy.backend isa CPU
        _sync_finest_matrix!(hierarchy, A)
        refresh_hierarchy!(hierarchy, solver)
        _amg_mixed_precision(hierarchy) && _sync_storage_levels!(hierarchy)
    else
        # Finest level refreshed entirely on device: nzval D2D + device diag + device lambda_max.
        # No D->H copy of the finest nzval, no host power iteration, no H->D recopy of level 1.
        _refresh_finest_level_device!(hierarchy, A)
        _refresh_finest_lambda_device!(hierarchy)
        # Coarse operators: device-resident RAP on CUDA, host RAP + sync on other backends.
        _refresh_coarse_operators!(hierarchy, solver)
    end
    workspace.refresh_count += 1
    return workspace
end

function solve_system!(phiEqn::ModelEquation, setup::SolverSetup{F,I,S1,S2,PT}, result, component, config) where {F,I,S1<:AMG,S2,PT}
    (; itmax, atol, rtol, solver) = setup
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    workspace = phiEqn.solver
    values = get_values(result, component)
    update!(workspace, A, solver, config)
    apply_smoother!(setup.smoother, values, A, b, config.hardware)
    x = workspace.solution
    copyto!(x, values)

    # Outer Krylov/defect-correction runs at working precision. Default (TS==TW): the finest
    # hierarchy operator (unchanged). Mixed precision: the FP32 finest cannot carry the outer
    # residual, so use the raw FP64 system matrix A (its _matvec!/_residual! are backend-dispatched).
    outer_A = (_greenfield_active(workspace.hierarchy) || _amg_mixed_precision(workspace.hierarchy)) ?
              A : workspace.hierarchy.levels[1].A
    _amg_solve_mode!(workspace, workspace.hierarchy, solver, solver.mode, outer_A, b, x; itmax=itmax, atol=atol, rtol=rtol)

    if typeof(phiEqn.model.terms[1].type) <: Time{CrankNicolson}
        xcal_foreach(x, config) do i
            x[i] = 2 * x[i] - values[i]
        end
    end

    copyto!(values, x)
    workspace.iterations == itmax && @warn "Maximum number of iterations reached!"
    return residual(phiEqn, component, config)
end

function _amg_solve_mode!(workspace, hierarchy, solver::AMG, ::AMGSolver, A, b, x; itmax, atol, rtol)
    return amg_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
end

function _amg_solve_mode!(workspace, hierarchy, solver::AMG, ::Cg, A, b, x; itmax, atol, rtol)
    return amg_cg_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
end
