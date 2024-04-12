export set_solver, set_runtime
export explicit_relaxation!, implicit_relaxation!, setReference!
export run!

set_solver( field::AbstractField; # To do - relax inputs and correct internally
    solver::S, 
    preconditioner::PT, 
    convergence, 
    relax, 
    itmax::Integer=100, 
    atol=sqrt(eps(_get_float(field.mesh))),
    rtol=_get_float(field.mesh)(1e-3)
    ) where {S,PT<:PreconditionerType} = 
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

function run!(phiEqn::ModelEquation, setup, result) # ; opP, solver

    # (; itmax, atol, rtol) = setup
    (; A, b) = phiEqn.equation
    precon = phiEqn.preconditioner
    # (; P) = precon 
    solver = phiEqn.solver
    (; x) = solver
    # values_eqn = get_phi(phiEqn).values
    # values_res = result.values
    (; values) = result

    backend = _get_backend(get_phi(phiEqn).mesh)

    _solve!(solver, A, b, values; setup, precon)

    # solve!(
    #     # solver, LinearOperator(A), b, values; M=P, itmax=itmax, atol=atol, rtol=rtol
    #     solver, A, b, values; M=P, itmax=itmax, atol=atol, rtol=rtol
    #     )
    KernelAbstractions.synchronize(backend)
    # gmres!(solver, A, b, values; M=P.P, itmax=itmax, atol=atol, rtol=rtol)
    # println(solver.stats.niter)
    kernel! = solve_copy_kernel!(backend)
    kernel!(values, x, ndrange = length(values))
    # kernel!(values_eqn, x, ndrange = length(values_eqn))
    KernelAbstractions.synchronize(backend)
    # kernel!(values_res, values_eqn, ndrange = length(values_eqn))
    # KernelAbstractions.synchronize(backend)
end

_solve!(solver, A, b, values; setup, precon) = begin
    (; itmax, atol, rtol) = setup
    solve!(solver, LinearOperator(A), b, values; M=precon.P, itmax=itmax, atol=atol, rtol=rtol)
end

_solve!(solver::QmrSolver, A, b, values; setup, precon) = begin
    (; itmax, atol, rtol) = setup
    solve!(solver, LinearOperator(A), b, values; itmax=itmax, atol=atol, rtol=rtol)
end

@kernel function solve_copy_kernel!(a, b)
    i = @index(Global)

    @inbounds begin
        a[i] = b[i]  
    end
end

# function explicit_relaxation!(phi, phi0, alpha)
#     @inbounds @simd for i ∈ eachindex(phi)
#         phi[i] = phi0[i] + alpha*(phi[i] - phi0[i])
#     end
# end

function explicit_relaxation!(phi, phi0, alpha)
    backend = _get_backend(phi.mesh)

    kernel! = explicit_relaxation_kernel!(backend)
    kernel!(phi, phi0, alpha, ndrange = length(phi))
    KernelAbstractions.synchronize(backend)
end

@kernel function explicit_relaxation_kernel!(phi, phi0, alpha)
    i = @index(Global)

    @inbounds begin
        phi[i] = phi0[i] + alpha*(phi[i] - phi0[i])
    end
end

# function implicit_relaxation!(eqn::E, field, alpha) where E<:Equation
#     (; A, b) = eqn
#     @inbounds for i ∈ eachindex(b)
#         A[i,i] /= alpha
#         b[i] += (1.0 - alpha)*A[i,i]*field[i]
#     end
# end

## IMPLICIT RELAXATION KERNEL 

# Prepare variables for kernel and call
function implicit_relaxation!(eqn::E, field, alpha, mesh) where E<:ModelEquation
    (; A, b) = eqn.equation
    precon = eqn.preconditioner
    # Output sparse matrix properties and values
    # rowval, colptr, nzval = sparse_array_deconstructor(A)
    rowval_array = _rowval(A)
    colptr_array = _colptr(A)
    nzval_array = _nzval(A)

    # Get backend and define kernel
    backend = _get_backend(mesh)
    kernel! = implicit_relaxation_kernel!(backend)
    
    # Define variable equal to 1 with same type as mesh integers
    integer = _get_int(mesh)
    ione = one(integer)
    
    # Execute kernel
    kernel!(ione, rowval_array, colptr_array, nzval_array, b, field, alpha, ndrange = length(b))
    KernelAbstractions.synchronize(backend)

    # check_for_precon!(nzval_array, precon, backend)
end

@kernel function implicit_relaxation_kernel!(ione, rowval, colptr, nzval, b, field, alpha)
    # i defined as values from 1 to length(b)
    i = @index(Global)
    
    @inbounds begin

        # Find nzval index relating to A[i,i] (CHANGE TO WHILE LOOP, WRAP IN FUNCTION)
        start = colptr[i]
        offset = 0
        for j in start:length(rowval)
            offset += 1
            if rowval[j] == i
                break
            end
        end
        nIndex = start + offset - ione

        # Run implicit relaxation calculations
        nzval[nIndex] /= alpha
        b[i] += (1.0 - alpha)*nzval[nIndex]*field[i]
    end
end

# function setReference!(pEqn::E, pRef, cellID) where E<:Equation
#     if pRef === nothing
#         return nothing
#     else
#         pEqn.b[cellID] += pEqn.A[cellID,cellID]*pRef
#         pEqn.A[cellID,cellID] += pEqn.A[cellID,cellID]
#     end
# end

function setReference!(pEqn::E, pRef, cellID) where E<:ModelEquation
    if pRef === nothing
        return nothing
    else
        backend = _get_backend((get_phi(pEqn)).mesh)
        ione = one(_get_int((get_phi(pEqn)).mesh))
        (; b, A) = pEqn.equation
        nzval_array = nzval(A)
        colptr_array = colptr(A)
        rowval_array = rowval(A)

        kernel! = setReference_kernel!(backend)
        kernel!(nzval_array, colptr_array, rowval_array, b, pRef, ione, cellID, ndrange = 1)
        KernelAbstractions.synchronize(backend)

    end
end

@kernel function setReference_kernel!(nzval, colptr, rowval, b, pRef, ione, cellID)
    i = @index(Global)

    @inbounds begin
        nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
        b[cellID] = nzval[nIndex]*pRef
        nzval[nIndex] += nzval[nIndex]
    end
end