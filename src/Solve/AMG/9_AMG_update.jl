function _pattern_matches(hierarchy::AMGHierarchy, A)
    hierarchy.nrows == _m(A) || return false
    hierarchy.nnz == length(_nzval(A)) || return false
    rp = _rowptr(A)
    cv = _colval(A)
    rp === hierarchy.rowptr_ref[] && cv === hierarchy.colval_ref[] && return true
    rowptr = _cpu_vector(rp)
    colval = _cpu_vector(cv)
    length(rowptr) == length(hierarchy.rowptr_pattern) || return false
    length(colval) == length(hierarchy.colval_pattern) || return false
    h = hash(colval, hash(rowptr))
    if h == hierarchy.pattern_hash
        hierarchy.rowptr_ref[] = rp
        hierarchy.colval_ref[] = cv
        return true
    end
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
    return _amg_update!(hierarchy, workspace, A, solver, hardware)
end

function _amg_update!(hierarchy::AMGHierarchy, workspace::AMGWorkspace, A, solver::AMG, hardware)
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
        _refresh_finest_level_device!(hierarchy, A)
        _refresh_finest_lambda_device!(hierarchy)
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

    outer_A = outer_operator(workspace.hierarchy, A)
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

# Mixed precision / matrix-free: FP32 finest cannot carry the outer residual, so use raw system matrix
outer_operator(hierarchy::AMGHierarchy, A) = _amg_mixed_precision(hierarchy) ? A : hierarchy.levels[1].A
outer_operator(::MatrixFreeHierarchy, A) = A

# NEW SECTION

@kernel function _amg_gather_kernel!(out, @Const(src), @Const(perm))
    k = @index(Global)
    @inbounds out[k] = src[perm[k]]
end

@kernel function _amg_scatter_kernel!(out, @Const(src), @Const(perm))
    k = @index(Global)
    @inbounds out[perm[k]] = src[k]
end

_matrix_free_coarse_max_rows(cs::OnDevice) = cs.max_rows
_matrix_free_coarse_max_rows(::Any) = 512

_matrix_free_omega(s::AMGJacobi) = s.omega
_matrix_free_omega(::Any) = 4 / 3

# NEW SECTION

function _build_matrix_free_workspace!(workspace::AMGWorkspace, A, solver::AMG, hardware)
    backend = hardware.backend
    setup_matrix = _amg_setup_matrix(A, _amg_setup_backend(backend))  # host CSR at finest precision T
    T = eltype(_nzval(_amg_matrix(setup_matrix)))
    TS = _effective_storage(T, _amg_storage(solver.coarse_storage))
    fused_top = max(Int(solver.fuse_levels) - 1, 0)
    handle = build_matrix_free_hierarchy(setup_matrix, solver.coarsening.merge_levels, backend;
                          pre=solver.pre_sweeps, post=solver.post_sweeps,
                          omega_nominal=_matrix_free_omega(solver.smoother),
                          max_coarse=solver.max_coarse_rows, fused_top=fused_top,
                          coarse_max_rows=_matrix_free_coarse_max_rows(solver.coarse_solve),
                          scale_correction=solver.scale_correction,
                          coarse_storage=TS)
    st = handle.st
    st.refresh_plan[] = build_refresh_plan(handle)
    st.workgroup = workspace.hierarchy.workgroup
    _amg_log_hierarchy(solver, backend, vcat([lv.n for lv in st.levels], st.coarse_n); matrix_free=true)
    workspace.hierarchy = st
    return workspace
end

function _amg_update!(hierarchy::MatrixFreeHierarchy, workspace::AMGWorkspace, A, solver::AMG, hardware)
    if isempty(hierarchy.levels) || hierarchy.nrows != _m(A) || hierarchy.nnz != length(_nzval(A))
        return _build_matrix_free_workspace!(workspace, A, solver, hardware)
    end
    refresh_coarse = (workspace.refresh_count + 1) % solver.coarse_refresh_interval == 0
    refresh_matrix_free_hierarchy!(hierarchy, hierarchy.refresh_plan[], A;
                                   omega_nominal=_matrix_free_omega(solver.smoother), coarse=refresh_coarse)
    workspace.refresh_count += 1
    return workspace
end

# NEW SECTION

function amg_apply_preconditioner!(z, hierarchy::MatrixFreeHierarchy, solver::AMG, r)
    bk = hierarchy.backend; wg = hierarchy.workgroup; n = hierarchy.nrows
    _launch_amg_kernel!(bk, wg, _amg_gather_kernel!, n, hierarchy.residual_permuted, r, hierarchy.cell_perm_device)
    x_perm = matrix_free_cycle!(hierarchy, hierarchy.residual_permuted)  # synchronizes before returning
    _launch_amg_kernel!(bk, wg, _amg_scatter_kernel!, n, z, x_perm, hierarchy.cell_perm_device)
    return z
end
