export set_solver, set_runtime
export explicit_relaxation!, implicit_relaxation!, implicit_relaxation_diagdom!, setReference!
export solve_system!
export solve_equation!
export residual!

"""
    set_solver( 
        field::AbstractField;
        # keyword arguments and defaults
        solver::S, 
        preconditioner::PT, 
        convergence, 
        relax,
        smoother=nothing,
        limit=(),
        itmax::Integer=1000, 
        atol=(eps(_get_float(field.mesh)))^0.9,
        rtol=_get_float(field.mesh)(1e-1)
        ) where {S,PT<:PreconditionerType} = begin

        # return NamedTuple
        TF = _get_float(field.mesh)
        (
            solver=solver, 
            preconditioner=preconditioner, 
            convergence=convergence |> TF, 
            relax=relax |> TF, 
            smoother=smoother,
            limit=limit,
            itmax=itmax, 
            atol=atol |> TF, 
            rtol=rtol |> TF
        )
    end

This function is used to provide solver settings that will be used internally in XCALibre.jl. It returns a `NamedTuple` with solver settings that are used internally by the flow solvers. 

# Input arguments

- `field` reference to the field to which the solver settings will apply (used to provide integer and float types required)
- `solver` solver object from Krylov.jl and it could be one of `BicgstabSolver`, `CgSolver`, `GmresSolver` which are re-exported in XCALibre.jl
- `preconditioner` instance of preconditioner to be used e.g. Jacobi()
- `convergence` sets the stopping criteria of this field
- `relax` specifies the relaxation factor to be used e.g. set to 1 for no relaxation
- `smoother` specifies smoothing method to be applied before discretisation. `JacobiSmoother` is currently the only choice (defaults to `nothing`)
- `limit` used in some solvers to bound the solution within this limits e.g. (min, max). It defaults to `()`
- `itmax` maximum number of iterations in a single solver pass (defaults to 1000) 
- `atol` absolute tolerance for the solver (default to eps(FloatType)^0.9)
- `rtol` set relative tolerance for the solver (defaults to 1e-1)
"""
set_solver( 
        field::AbstractField;
        # keyword arguments and defaults
        solver::S, 
        preconditioner::PT, 
        convergence, 
        relax,
        smoother=nothing,
        limit=(),
        itmax::Integer=1000, 
        atol=(eps(_get_float(field.mesh)))^0.9,
        rtol=_get_float(field.mesh)(1e-1)
    ) where {S,PT<:PreconditionerType} = begin

    # return NamedTuple
    TF = _get_float(field.mesh)
    (
        solver=solver, 
        preconditioner=preconditioner, 
        convergence=convergence |> TF, 
        relax=relax |> TF, 
        smoother=smoother,
        limit=limit,
        itmax=itmax, 
        atol=atol |> TF, 
        rtol=rtol |> TF
    )
end


"""
    set_runtime(; 
        # keyword arguments
        iterations::I, 
        write_interval::I, 
        time_step::N
        ) where {I<:Integer,N<:Number} = begin
        
        # returned `NamedTuple`
            (
            iterations=iterations, 
            dt=time_step, 
            write_interval=write_interval)
    end

This is a convenience function to set the top-level runtime information. The inputs are all keyword arguments and provide basic information to flow solvers just before running a simulation.

# Input arguments

* `iterations::Integer` specifies the number of iterations in a simulation run.
* `write_interval::Integer` defines how often simulation results are written to file (on the current working directory). The interval is currently based on number of iterations. Set to `-1` to run without writing results to file.
* `time_step::Number` the time step to use in the simulation. Notice that for steady solvers this is simply a counter and it is recommended to simply use `1`.

# Example

```julia
runtime = set_runtime(
    iterations=2000, time_step=1, write_interval=2000)
```
"""
set_runtime(; iterations::I, write_interval::I, time_step::N) where {I<:Integer,N<:Number} = begin
    (iterations=iterations, dt=float(time_step), write_interval=write_interval)
end

function solve_equation!(
    eqn::ModelEquation{T,M,E,S,P}, phi, solversetup, config; time=nothing, ref=nothing, irelax=nothing
    ) where {T<:ScalarModel,M,E,S,P}

    discretise!(eqn, phi, config)       
    apply_boundary_conditions!(eqn, phi.BCs, nothing, time, config)
    setReference!(eqn, ref, 1, config)
    if !isnothing(irelax)
        implicit_relaxation!(eqn, phi.values, irelax, nothing, config)
        # implicit_relaxation_diagdom!(eqn, phi.values, irelax, nothing, config)
    end
    update_preconditioner!(eqn.preconditioner, phi.mesh, config)
    res = solve_system!(eqn, solversetup, phi, nothing, config)
    return res
end

function solve_equation!(
    psiEqn::ModelEquation{T,M,E,S,P}, psi, solversetup, xdir, ydir, zdir, config; time=nothing
    ) where {T<:VectorModel,M,E,S,P}

    mesh = psi.mesh

    discretise!(psiEqn, psi, config)
    update_equation!(psiEqn, config)

    apply_boundary_conditions!(psiEqn, psi.x.BCs, xdir, time, config)
    # implicit_relaxation!(psiEqn, psi.x.values, solversetup.relax, xdir, config)
    implicit_relaxation_diagdom!(psiEqn, psi.x.values, solversetup.relax, xdir, config)
    update_preconditioner!(psiEqn.preconditioner, mesh, config)
    resx = solve_system!(psiEqn, solversetup, psi.x, xdir, config)
    
    update_equation!(psiEqn, config)
    apply_boundary_conditions!(psiEqn, psi.y.BCs, ydir, time, config)
    # implicit_relaxation!(psiEqn, psi.y.values, solversetup.relax, ydir, config)
    implicit_relaxation_diagdom!(psiEqn, psi.y.values, solversetup.relax, ydir, config)
    update_preconditioner!(psiEqn.preconditioner, mesh, config)
    resy = solve_system!(psiEqn, solversetup, psi.y, ydir, config)
    
    # Z velocity calculations (3D Mesh only)
    resz = one(_get_float(mesh))
    if typeof(mesh) <: Mesh3
        update_equation!(psiEqn, config)
        apply_boundary_conditions!(psiEqn, psi.z.BCs, zdir, time, config)
        # implicit_relaxation!(psiEqn, psi.z.values, solversetup.relax, zdir, config)
        implicit_relaxation_diagdom!(psiEqn, psi.z.values, solversetup.relax, zdir, config)
        update_preconditioner!(psiEqn.preconditioner, mesh, config)
        resz = solve_system!(psiEqn, solversetup, psi.z, zdir, config)
    end
    return resx, resy, resz
end

function solve_system!(phiEqn::ModelEquation, setup, result, component, config) # ; opP, solver

    (; itmax, atol, rtol) = setup
    precon = phiEqn.preconditioner
    (; P) = precon 
    solver = phiEqn.solver
    (; x) = solver
    
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; values, mesh) = result
    
    A = _A(phiEqn)
    # opA = phiEqn.equation.opA
    opA = A
    b = _b(phiEqn, component)

    apply_smoother!(setup.smoother, values, A, b, hardware)

    solve!(
        solver, opA, b, values; 
        M=P, itmax=itmax, atol=atol, rtol=rtol, ldiv=is_ldiv(precon)
        )
    # KernelAbstractions.synchronize(backend)

    statistics(solver).niter == itmax && @warn "Maximum number of iteration reached!"

    # println(statistics(solver).niter)
    
    kernel! = _copy!(backend, workgroup)
    kernel!(values, x, ndrange = length(values))
    # KernelAbstractions.synchronize(backend)

    res = residual(phiEqn, component, config)
    return res
end

@kernel function _copy!(a, b)
    i = @index(Global)

    @inbounds begin
        a[i] = b[i]  
    end
end

function explicit_relaxation!(phi, phi0, alpha, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel! = explicit_relaxation_kernel!(backend, workgroup)
    kernel!(phi, phi0, alpha, ndrange = length(phi))
    # KernelAbstractions.synchronize(backend)
end

@kernel function explicit_relaxation_kernel!(phi, phi0, alpha)
    i = @index(Global)

    @inbounds begin
        phi[i] = phi0[i] + alpha*(phi[i] - phi0[i])
    end
end

## IMPLICIT RELAXATION KERNEL 

# Prepare variables for kernel and call
function implicit_relaxation!(
    phiEqn::E, field, alpha, component, config) where E<:ModelEquation
    mesh = get_phi(phiEqn).mesh
    (; hardware) = config
    (; backend, workgroup) = hardware
    precon = phiEqn.preconditioner
    # Output sparse matrix properties and values
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    rowval_array = _colval(A)
    colptr_array = _rowptr(A)
    nzval_array = _nzval(A)

    # Get backend and define kernel
    kernel! = implicit_relaxation_kernel!(backend, workgroup)
    
    # Define variable equal to 1 with same type as mesh integers
    integer = _get_int(mesh)
    ione = one(integer)
    
    # Execute kernel
    kernel!(ione, rowval_array, colptr_array, nzval_array, b, field, alpha, ndrange = length(b))
    # KernelAbstractions.synchronize(backend)

    # check_for_precon!(nzval_array, precon, backend)
end

@kernel function implicit_relaxation_kernel!(ione, colval, rowptr, nzval, b, field, alpha)
    # i defined as values from 1 to length(b)
    i = @index(Global)
    
    @inbounds begin

        # Find nzval index relating to A[i,i]
        nIndex = spindex(rowptr, colval, i, i)

        # Run implicit relaxation calculations
        nzval[nIndex] /= alpha
        b[i] += (1.0 - alpha)*nzval[nIndex]*field[i]
    end
end


## IMPLICIT RELAXATION KERNEL with DIAGONAL DOMINANCE

# Prepare variables for kernel and call
function implicit_relaxation_diagdom!(
    phiEqn::E, field, alpha, component, config) where E<:ModelEquation
    mesh = get_phi(phiEqn).mesh
    (; cells, cell_neighbours) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware
    precon = phiEqn.preconditioner
    # Output sparse matrix properties and values
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    rowval_array = _colval(A)
    colptr_array = _rowptr(A)
    nzval_array = _nzval(A)

    # Get backend and define kernel
    kernel! = _implicit_relaxation_diagdom!(backend, workgroup)
    
    # Define variable equal to 1 with same type as mesh integers
    integer = _get_int(mesh)
    ione = one(integer)
    
    # Execute kernel
    kernel!(cells, cell_neighbours, ione, rowval_array, colptr_array,
    nzval_array, b, field, alpha, ndrange = length(b))
    # KernelAbstractions.synchronize(backend)
end

@kernel function _implicit_relaxation_diagdom!(cells::AbstractArray{Cell{TF,SV,UR}}, cell_neighbours, 
    ione, colval, rowptr, nzval, b, field, alpha) where {TF,SV,UR}
    # i defined as values from 1 to length(b)
    i = @index(Global)
    
    sumv = zero(TF)

    @inbounds begin

        # Find nzval index relating to A[i,i]
        cIndex = spindex(rowptr, colval, i, i)
        
        # (; faces_range) = cells[i]
        # for ni ∈ faces_range
        #     nID = cell_neighbours[ni]
        #     zIndex = spindex(rowptr, colval, i, nID)
        #     sumv += abs(nzval[zIndex])
        # end

        start_index = rowptr[i]
        end_index = rowptr[i+1] -1
        for nzi ∈ start_index:end_index
            sumv += abs(nzval[nzi])
        end
        sumv -= abs(nzval[cIndex]) # remove diagonal contribution

        # Run implicit relaxation calculations
        D0 = nzval[cIndex]
        D_max = max(abs(D0), sumv)/alpha
        nzval[cIndex] = D_max
        b[i] += (D_max - D0)*field[i]
    end
end


function setReference!(pEqn::E, pRef, cellID, config) where E<:ModelEquation
    if pRef === nothing
        return nothing
    else
        (; hardware) = config
        (; backend, workgroup) = hardware
        ione = one(_get_int((get_phi(pEqn)).mesh))
        (; b, A) = pEqn.equation
        nzval_array = nzval(A)
        colptr_array = rowptr(A)
        rowval_array = colval(A)

        kernel! = _setReference!(backend, workgroup)
        kernel!(nzval_array, colptr_array, rowval_array, b, pRef, ione, cellID, ndrange = 1)
        # KernelAbstractions.synchronize(backend)

    end
end

@kernel function _setReference!(nzval, rowptr, colval, b, pRef, ione, cellID)
    i = @index(Global)

    @inbounds begin
        nIndex = spindex(rowptr, colval, cellID, cellID)
        b[cellID] = nzval[nIndex]*pRef
        nzval[nIndex] += nzval[nIndex]
    end
end

function residual(eqn, component, config)
    (; A, R, Fx) = eqn.equation
    b = _b(eqn, component)
    values = get_values(get_phi(eqn), component)

    # # Openfoam's residual definition (not optimised)
    # Fx .= A*values
    # R .= mean(values)
    # Fx_mean = A*R 
    # T1 = mean(norm.(b .- Fx))
    # T2 = mean(norm.(Fx .- Fx_mean))
    # T3 = mean(norm.(b .- Fx_mean))
    # Residual = T1/(T2 + T3)

    # Previous definition
    Fx .= A * values
    @inbounds @. R = (b - Fx)^2
    normb = norm(b)
    denominator = ifelse(normb>0,normb, 1)
    Residual = sqrt(mean(R)) / denominator
    return Residual
end