export set_solver, set_runtime
export explicit_relaxation!, implicit_relaxation!, setReference!
export run!

set_solver( field::AbstractField; # To do - relax inputs and correct internally
    solver::S, 
    preconditioner::PT, 
    convergence::F, 
    relax::F, 
    itmax::I=100, 
    atol::F=sqrt(eps()),
    rtol::F=1e-3 
    ) where {S,PT<:PreconditionerType,I<:Integer,F<:AbstractFloat} = 
begin
    TF = _get_float(field.mesh)
    # TI = _get_int(field.mesh)
    (
        solver=solver, 
        preconditioner=preconditioner, 
        convergence=convergence |> TF, 
        relax=relax |> TF, 
        itmax=itmax, 
        atol=atol |> TF, 
        rtol=rtol |> TF
    )
end

set_runtime(; iterations::I, write_interval::I, time_step::N) where {I<:Integer,N<:Number} = begin
    (iterations=iterations, dt=time_step, write_interval=write_interval)
end

function run!(phiEqn::ModelEquation, setup) # ; opP, solver

    (; itmax, atol, rtol) = setup
    (; A, b) = phiEqn.equation
    P = phiEqn.preconditioner
    solver = phiEqn.solver
    values = get_phi(phiEqn).values

    solve!(
        solver, A, b, values; M=P.P, itmax=itmax, atol=atol, rtol=rtol
        )
    # println(solver.stats.niter)
    @turbo values .= solver.x
    nothing
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