export SolverSetup
export run!
# export residual, residual_print
# export relax!, update_residual!, update_solution!
# export clear!

struct SolverSetup{S,I,F}
    solver::S 
    # iterations::I 
    # tolerance::F 
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
    # SolverSetup{S,I,F}(solver, iterations, tolerance, relax, itmax, atol, rtol)
    SolverSetup{S,I,F}(solver, relax, itmax, atol, rtol)
end

function run!(
    equation::Equation{Ti,Tf}, phiModel, BCs, setup; 
    # correct_term=nothing, opA, opP, solver
    opA, opP, solver
    ) where {Ti,Tf}
    
    @inbounds equation.b .+= phiModel.sources[1].field # should be moved out to "add_sources" function using the "Model" struct

    (; relax, itmax, atol, rtol) = setup
    (; A, b, R, Fx) = equation
    (; phi) = phiModel.terms[1]
    (; values, mesh) = phi

    solve!(
        solver, opA, b, values; 
        M=opP, itmax=itmax, atol=atol, rtol=rtol
        )
    # relax!(values, solver.x, 1.0)
    # println(solver.stats.niter)
    @turbo values .= solver.x

end