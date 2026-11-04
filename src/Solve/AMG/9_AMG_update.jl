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

# `mesh` gives Pairwise() its face-normal pairing bias; without it equal couplings pair in cell order
function update!(workspace::AMGWorkspace, A, solver::AMG, config, mesh=nothing)
    (; hardware) = config
    hierarchy = workspace.hierarchy
    hierarchy.backend = hardware.backend
    hierarchy.workgroup = _amg_workgroup(hardware.backend, hardware.workgroup, max(_m(A), 1))
    return _amg_update!(hierarchy, workspace, A, solver, hardware, mesh)
end

function _amg_update!(hierarchy::AMGHierarchy, workspace::AMGWorkspace, A, solver::AMG, hardware, _mesh)
    if isempty(hierarchy.host_levels)
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        workspace.hierarchy = setup_hierarchy(setup_matrix, solver, hardware.backend, hardware.workgroup;
            log_diagnostics=true, index_type=eltype(hierarchy.rowptr_pattern))
        return workspace
    end

    if !_pattern_matches(hierarchy, A)
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        workspace.hierarchy = setup_hierarchy(setup_matrix, solver, hardware.backend, hardware.workgroup;
            log_diagnostics=false, index_type=eltype(hierarchy.rowptr_pattern))
        return workspace
    end

    numeric_updates = workspace.refresh_count
    refresh_coarse = (numeric_updates + 1) % solver.refresh == 0
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

function solve_system!(phiEqn::ModelEquation, setup::SolverSetup{F,I,S1,S2,<:AMG}, result, component, config) where {F,I,S1,S2}
    (; itmax, atol, rtol) = setup
    solver = setup.preconditioner
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    workspace = phiEqn.solver
    values = result.values
    res = residual(phiEqn, component, config) # outer residual of x₀: independent of the solve
    update!(workspace, A, solver, config, get_phi(phiEqn).mesh)
    apply_smoother!(setup.smoother, values, A, b, config.hardware)
    x = workspace.solution
    copyto!(x, values)

    outer_A = outer_operator(workspace.hierarchy, A)
    # 1/|diag(A)| for the stopping norm ‖D⁻¹r‖, in Fx (scratch once `residual` has run)
    (; Fx, diag_nz) = phiEqn.equation
    _inv_abs_diagonal!(Fx, _nzval(A), diag_nz)
    _amg_solve_mode!(workspace, workspace.hierarchy, solver, setup.solver, outer_A, b, x;
        itmax=itmax, atol=atol, rtol=rtol, dscale=Fx)

    copyto!(values, x)
    workspace.converged || _warn_unconverged(phiEqn, setup, component, _amg_reason(workspace, itmax),
        workspace.iterations, config)
    return res
end

_amg_reason(workspace, itmax) = workspace.iterations ≥ itmax ?
    "maximum number of iterations exceeded" : "stagnating, non-finite or indefinite iteration"

function _amg_solve_mode!(workspace, hierarchy, solver::AMG, ::Richardson, A, b, x; itmax, atol, rtol, dscale=nothing)
    return amg_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol, dscale=dscale)
end

function _amg_solve_mode!(workspace, hierarchy, solver::AMG, ::Bicgstab, A, b, x; itmax, atol, rtol, dscale=nothing)
    return amg_bicgstab_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol, dscale=dscale)
end

function _amg_solve_mode!(workspace, hierarchy, solver::AMG, ::Cg, A, b, x; itmax, atol, rtol, dscale=nothing)
    return amg_cg_solve!(workspace, hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol, dscale=dscale)
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

_matrix_free_omega(s::Jacobi) = s.weight
_matrix_free_omega(s) = 4 / 3

# NEW SECTION

function _build_matrix_free_workspace!(workspace::AMGWorkspace, A, solver::AMG, hardware, mesh)
    backend = hardware.backend
    _check_smoother_backend(backend, solver.smoother)
    setup_matrix = _amg_setup_matrix(A, _amg_setup_backend(backend))  # host CSR at finest precision T
    T = eltype(_nzval(_amg_matrix(setup_matrix)))
    TS = _effective_storage(T, _amg_storage(solver))
    coarse = solver.native.coarse
    handle = build_matrix_free_hierarchy(setup_matrix, solver.coarsening.merges, backend;
                          sweeps=solver.sweeps, smoother=solver.smoother,
                          omega_nominal=_matrix_free_omega(solver.smoother),
                          max_coarse=solver.coarsest, fused_top=solver.native.fuse,
                          coarse_max_rows=coarse isa Direct ? coarse.max_rows : 0,
                          scale_correction=solver.native.scale_correction,
                          coarse_storage=TS, mesh)
    st = handle.st
    st.refresh_plan[] = build_refresh_plan(handle)
    st.coarse_inv[] = _mf_coarse_solver(coarse, st, st.refresh_plan[].coarsest_csr, backend)
    st.workgroup = workspace.hierarchy.workgroup
    _amg_log_hierarchy(solver, backend, vcat([lv.n for lv in st.levels], st.coarse_n); matrix_free=true)
    workspace.hierarchy = st
    return workspace
end

function _amg_update!(hierarchy::MatrixFreeHierarchy, workspace::AMGWorkspace, A, solver::AMG, hardware, mesh)
    if isempty(hierarchy.levels) || hierarchy.nrows != _m(A) || hierarchy.nnz != length(_nzval(A))
        return _build_matrix_free_workspace!(workspace, A, solver, hardware, mesh)
    end
    refresh_coarse = (workspace.refresh_count + 1) % solver.refresh == 0
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
