export SolverSetup, solver_setup
export run!

struct SolverSetup{S,I,F}
    solver::S 
    relax::F 
    itmax::I
    atol::F 
    rtol::F
end
SolverSetup(
    ; solver::S, 
    # iterations::I, 
    # tolerance::F, 
    relax::F, 
    itmax::I=200, 
    atol::F=1e-15, 
    rtol::F=1e-1
    ) where {S,I,F} = begin
    SolverSetup{S,I,F}(solver, relax, itmax, atol, rtol)
end

solver_setup(field; ) = begin
    nothing
end

function run!(phiModel::Model, setup; opP, solver)

    (; itmax, atol, rtol) = setup
    (; A, b) = phiModel.equation
    phi = get_phi(phiModel)
    values = phi.values

    solve!(
        solver, A, b, values; 
        M=opP, itmax=itmax, atol=atol, rtol=rtol
        )
    # println(solver.stats.niter)
    @turbo values .= solver.x

end