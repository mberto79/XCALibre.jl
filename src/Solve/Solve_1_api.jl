export set_solver
export explicit_relaxation!, implicit_relaxation!, setReference!
export run!

set_solver( field::AbstractField; # To do - relax inputs and correct internally
    solver::S, 
    preconditioner::PT, 
    P::PP=nothing,
    convergence::F, 
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
        convergence=convergence, 
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

function explicit_relaxation!(phi, phi0, alpha)
    @inbounds @simd for i ∈ eachindex(phi)
        phi[i] = phi0[i] + alpha*(phi[i] - phi0[i])
    end
end

function implicit_relaxation!(eqn::E, field, alpha) where E<:Equation
    (; A, b) = eqn
    @inbounds for i ∈ eachindex(b)
        A[i,i] /= alpha
        b[i] += (1.0 - alpha)*A[i,i]*field[i]
    end
end

function setReference!(pEqn::E, pRef, cellID) where E<:Equation
    if pRef === nothing
        return nothing
    else
        pEqn.b[cellID] += pEqn.A[cellID,cellID]*pRef
        pEqn.A[cellID,cellID] += pEqn.A[cellID,cellID]
    end
end