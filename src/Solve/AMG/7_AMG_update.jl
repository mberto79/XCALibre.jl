function update!(workspace::AMGWorkspace, A, solver::AMG, config)
    if isnothing(workspace.hierarchy)
        (; hardware) = config
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        workspace.hierarchy = setup_hierarchy(setup_matrix, solver, setup_backend)
        return workspace
    end

    hierarchy = workspace.hierarchy
    if hierarchy.nrows != _m(A) || hierarchy.nnz != length(_nzval(A))
        (; hardware) = config
        setup_backend = _amg_setup_backend(hardware.backend)
        setup_matrix = _amg_setup_matrix(A, setup_backend)
        workspace.hierarchy = setup_hierarchy(setup_matrix, solver, setup_backend)
        return workspace
    end

    refresh_hierarchy!(hierarchy, solver)
    return workspace
end

function solve_system!(phiEqn::ModelEquation, setup::SolverSetup{F,I,S1,S2,PT}, result, component, config) where {F,I,S1<:AMG,S2,PT}
    (; itmax, atol, rtol, solver) = setup
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    x = get_values(result, component)
    workspace = phiEqn.solver
    update!(workspace, A, solver, config)

    if solver.mode == :solver
        amg_solve!(workspace, workspace.hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    else
        amg_cg_solve!(workspace, workspace.hierarchy, solver, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
    end

    return residual(phiEqn, component, config)
end
