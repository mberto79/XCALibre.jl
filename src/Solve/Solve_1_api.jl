export SolverSetup, Runtime, Schemes
export explicit_relaxation!, implicit_relaxation!, implicit_relaxation_diagdom!, setReference!
export solve_system!
export solve_equation!
export residual!

struct SolverSetup{
    F<:AbstractFloat,
    I<:Integer,
    S1<:AbstractLinearSolver,
    S2<:Union{Nothing, AbstractSmoother},
    PT<:PreconditionerType
    }
    solver::S1
    smoother::S2
    preconditioner::PT
    convergence::F
    relax::F
    limit::Union{Nothing, Tuple{F,F}}
    itmax::I
    atol::F
    rtol::F
end

"""
    SolverSetup(; 
            # required keyword arguments 

            solver::S, 
            preconditioner::PT, 
            convergence, 
            relax,

            # optional keyword arguments

            float_type=Float64,
            smoother=nothing,
            limit=nothing,
            itmax::Integer=1000, 
            atol=(eps(_get_float(region)))^0.9,
            rtol=_get_float(region)(1e-1)

        ) where {S,PT<:PreconditionerType} = begin

            return SolverSetup(kwargs...)  
    end

This function is used to provide solver settings that will be used internally in XCALibre.jl. It returns a `SolverSetup` object with solver settings that are used internally by the flow solvers. 

# Input arguments

- `solver`: solver object from Krylov.jl and it could be one of `Bicgstab()`, `Cg()`, `Gmres()` which are re-exported in XCALibre.jl
- `preconditioner`: instance of preconditioner to be used e.g. Jacobi()
- `convergence` sets the stopping criteria of this field
- `relax`: specifies the relaxation factor to be used e.g. set to 1 for no relaxation
- `smoother`: specifies smoothing method to be applied before discretisation. `JacobiSmoother`: is currently the only choice (defaults to `nothing`)
- `limit`: used in some solvers to bound the solution within these limits e.g. (min, max). It defaults to `nothing`
- `itmax`: maximum number of iterations in a single solver pass (defaults to 1000) 
- `atol`: absolute tolerance for the solver (default to eps(FloatType)^0.9)
- `rtol`: set relative tolerance for the solver (defaults to 1e-1)
- `float_type`: specifies the floating point type to be used by the solver. It is also used to estimate the absolute tolerance for the solver (defaults to `Float64`)
"""
SolverSetup(;
        float_type=Float64,
        solver::S1, 
        smoother::S2=nothing,
        preconditioner::PT, 
        convergence, 
        relax, 
        limit=nothing,
        itmax::I=1000, 
        atol=(eps(float_type))^0.9,
        rtol=1e-1 |> float_type
        ) where{S1,S2,PT,I} = 
        SolverSetup{float_type,I,S1,S2,PT}(
            solver, smoother,preconditioner, 
            float_type(convergence), 
            float_type(relax), 
            limit,
            itmax, 
            float_type(atol),
            float_type(rtol))

struct Runtime{I<:Integer,F<:AbstractFloat}
    iterations::I
    dt::F
    write_interval::I
end

"""
    Runtime(; 
            # keyword arguments

            iterations::I, 
            write_interval::I, 
            time_step::N
        ) where {I<:Integer,N<:Number} = begin
        
        # returned Runtime struct
        Runtime{I<:Integer,F<:AbstractFloat}
            (
                iterations=iterations, 
                dt=time_step, 
                write_interval=write_interval
            )
    end

This is a convenience function to set the top-level runtime information. The inputs are all keyword arguments and provide basic information to flow solvers just before running a simulation.

# Input arguments

- `iterations::Integer`: specifies the number of iterations in a simulation run.
- `write_interval::Integer`: defines how often simulation results are written to file (on the current working directory). The interval is currently based on number of iterations. Set to `-1` to run without writing results to file.
- `time_step::AbstractFloat`: the time step to use in the simulation. Notice that for steady solvers this is simply a counter and it is recommended to simply use `1`.

# Example

```julia
runtime = Runtime(iterations=2000, time_step=1, write_interval=2000)
```
"""
Runtime(; iterations::I, write_interval::I, time_step::N) where {I<:Integer,N<:Number} = begin
    Runtime(iterations, float(time_step), write_interval)
end

# Set schemes function definition with default set variables
"""
    Schemes(;
        # keyword arguments and their default values
        time=SteadyState,
        divergence=Linear, 
        laplacian=Linear, 
        gradient=Gauss,
        limiter=nothing) = begin

        # Returns Schemes struct used to configure discretisation
        
        Schemes(
            time=time,
            divergence=divergence,
            laplacian=laplacian,
            gradient=gradient,
            limiter=limiter
        )   
    end

The `Schemes` struct is used at the top-level API to help users define discretisation schemes for every field solved.

# Inputs

- `time`: is used to set the time schemes (default is `SteadyState`)
- `divergence`: is used to set the divergence scheme (default is `Linear`) 
- `laplacian`: is used to set the laplacian scheme (default is `Linear`)
- `gradient`:  is used to set the gradient scheme (default is `Gauss`)
- `limiter`: is used to specify if gradient limiters should be used, currently supported limiters include `FaceBased` and `MFaceBased` (default is `nothing`)
"""
@kwdef struct Schemes
    time=SteadyState
    divergence=Linear
    laplacian=Linear
    gradient=Gauss
    limiter=nothing
end


function solve_equation!(
    eqn::ModelEquation{T,M,E,S,P}, phi, phiBCs, solversetup, config; time=nothing, ref=nothing, irelax=nothing
    ) where {T<:ScalarModel,M,E,S,P}

    discretise!(eqn, phi, config)       
    apply_boundary_conditions!(eqn, phiBCs, nothing, time, config)
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
    psiEqn::ModelEquation{T,M,E,S,P}, psi, psiBCs, solversetup, xdir, ydir, zdir, config; time=nothing
    ) where {T<:VectorModel,M,E,S,P}

    mesh = psi.mesh

    discretise!(psiEqn, psi, config)
    update_equation!(psiEqn, config)

    apply_boundary_conditions!(psiEqn, psiBCs, xdir, time, config)
    # implicit_relaxation!(psiEqn, psi.x.values, solversetup.relax, xdir, config)
    implicit_relaxation_diagdom!(psiEqn, psi.x.values, solversetup.relax, xdir, config)
    update_preconditioner!(psiEqn.preconditioner, mesh, config)
    resx = solve_system!(psiEqn, solversetup, psi.x, xdir, config)
    
    update_equation!(psiEqn, config)
    apply_boundary_conditions!(psiEqn, psiBCs, ydir, time, config)
    # implicit_relaxation!(psiEqn, psi.y.values, solversetup.relax, ydir, config)
    implicit_relaxation_diagdom!(psiEqn, psi.y.values, solversetup.relax, ydir, config)
    # update_preconditioner!(psiEqn.preconditioner, mesh, config)
    resy = solve_system!(psiEqn, solversetup, psi.y, ydir, config)
    
    # Z velocity calculations (3D Mesh only)
    # resz = one(_get_float(mesh))
    resz = zero(_get_float(mesh))
    if typeof(mesh) <: Mesh3
        update_equation!(psiEqn, config)
        apply_boundary_conditions!(psiEqn, psiBCs, zdir, time, config)
        # implicit_relaxation!(psiEqn, psi.z.values, solversetup.relax, zdir, config)
        implicit_relaxation_diagdom!(psiEqn, psi.z.values, solversetup.relax, zdir, config)
        # update_preconditioner!(psiEqn.preconditioner, mesh, config)
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

    krylov_solve!(
        solver, opA, b, values; 
        M=P, itmax=itmax, atol=atol, rtol=rtol, ldiv=is_ldiv(precon)
        )
    # KernelAbstractions.synchronize(backend)

    # Perform explicit step for Crank-Nicholson. Otherwise simply update field with solution
    if typeof(phiEqn.model.terms[1].type) <: Time{CrankNicolson}
        @. x = 2.0*x - values
    end

    ndrange = length(values)
    kernel! = _copy!(_setup(backend, workgroup, ndrange)...)
    kernel!(values, x)

    Krylov.iteration_count(solver) == itmax && @warn "Maximum number of iterations reached!"

    # println(statistics(solver).niter)
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

    ndrange = length(phi)
    kernel! = explicit_relaxation_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(phi, phi0, alpha)
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
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Extract sparse matrix properties and values
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    ndrange = length(b)
    kernel! = implicit_relaxation_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(colval, rowptr, nzval, b, field, alpha)
    # KernelAbstractions.synchronize(backend)
end

@kernel function implicit_relaxation_kernel!(colval, rowptr, nzval, b, field, alpha)
    i = @index(Global)
    
    @inbounds begin
        nIndex = spindex(rowptr, colval, i, i)
        nzval[nIndex] /= alpha
        b[i] += (1.0 - alpha)*nzval[nIndex]*field[i]
    end
end


## IMPLICIT RELAXATION KERNEL with DIAGONAL DOMINANCE

# Prepare variables for kernel and call
function implicit_relaxation_diagdom!(
    phiEqn::E, field, alpha, component, config) where E<:ModelEquation
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Extract sparse matrix properties and values
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    ndrange = length(b)
    kernel! = _implicit_relaxation_diagdom!(_setup(backend, workgroup, ndrange)...)
    kernel!(colval, rowptr, nzval, b, field, alpha)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _implicit_relaxation_diagdom!(colval, rowptr, nzval, b, field, alpha)
    i = @index(Global)
    
    sumv = zero(eltype(b))

    @inbounds begin

        # Find nzval index relating to A[i,i]
        cIndex = spindex(rowptr, colval, i, i)

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
        (; b, A) = pEqn.equation
        nzval = _nzval(A)
        colval = _colval(A)
        rowptr = _rowptr(A)

        ndrange = 1
        kernel! = _setReference!(_setup(backend, workgroup, ndrange)...)
        kernel!(nzval, colval, rowptr, b, pRef, cellID)
    end
end

@kernel function _setReference!(nzval, colval, rowptr, b, pRef, cellID)
    i = @index(Global)

    @inbounds begin
        cIndex = spindex(rowptr, colval, cellID, cellID)
        b[cellID] = nzval[cIndex]*pRef
        nzval[cIndex] += nzval[cIndex]
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