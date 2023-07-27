export SolverSetup
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

function run!(
    equation::Equation{Ti,Tf}, phiModel, BCs, setup; 
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
    # println(solver.stats.niter)
    @turbo values .= solver.x

end