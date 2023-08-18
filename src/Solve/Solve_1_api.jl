export SolverSetup, setup_solver
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

setup_solver( field::AbstractField; # To do - relax inputs and correct internally
    solver::S, 
    preconditioner::PT, 
    P::PP=nothing,
    tolerance::F, 
    relax::F, 
    itmax::I=100, 
    atol::F=sqrt(eps()),
    rtol::F=1e-3 
    ) where {S,PT<:PreconditionerType,PP,I<:Integer,F<:AbstractFloat} = 
begin
    teqn = Equation(field.mesh)
    (
        solver=solver(teqn.A,teqn.b), 
        preconditioner=preconditioner, 
        P=P,
        tolerance=tolerance, 
        relax=relax, 
        itmax=itmax, 
        atol=atol, 
        rtol=rtol
    )
end

function run!(phiModel::Model, setup) # ; opP, solver

    (; itmax, atol, rtol, P, solver) = setup
    (; A, b) = phiModel.equation
    phi = get_phi(phiModel)
    values = phi.values

    solve!(
        solver, A, b, values; 
        M=P.P, itmax=itmax, atol=atol, rtol=rtol
        )
    # println(solver.stats.niter)
    @turbo values .= solver.x

end