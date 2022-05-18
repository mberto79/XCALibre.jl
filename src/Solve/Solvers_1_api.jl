export run!, set_solver, residual, residual_print
export relax!, update_residual!, update_solution!

function set_solver(equation::Equation{I,F}, solver) where {I,F}
    solver(equation.A, equation.R)
end

function run!(
    solver, equation::Equation{Ti,Tf}, phi; iterations=100, tol=1e-6, alpha=1.0,
    atol=1e-12, rtol=1e-2, itmax=100, kwargs...
    ) where {Ti,Tf}
    (; A, b) = equation
    (; values) = phi
    F = ilu(A, τ = 0.005)
    
    # Definition of linear operators to reduce allocations during iterations
    opP = LinearOperator(Float64, A.m, A.n, false, false, (y, v) -> ldiv!(y, F, v))
    opA = LinearOperator(A)

    @inbounds for i ∈ 1:iterations
        solve!(
            solver, opA, b, values; 
            M=opP, itmax=itmax, atol=atol, rtol=rtol, kwargs...
            )
        relax!(phi, solver, alpha)
        update_residual!(equation, opA, phi)
        if residual(equation) <= tol
            residual_print(equation)
            println("Converged in ", i, " iterations")
            break
        end
    end
end

@inline function relax!(phi, solver, α)
    # phi.values .= (1.0 - α).*phi.values .+ α.*solver.x
    values = phi.values
    x = solver.x
    @inbounds for i ∈ eachindex(values)
        values[i] = (1.0 - α)*values[i] + α*x[i]
    end
end

@inline function update_solution!(phi, solver)
    relax!(phi, solver, 1.0)
end

@inline function update_residual!(equation, opA, phi::ScalarField{Ti,Tf}) where {Ti,Tf}
    (; b, R, Fx) = equation
    mul!(Fx, opA, phi.values)
    @inbounds for i ∈ eachindex(R)
        R[i] = b[i] - Fx[i]
    end
end

function residual(equation::Equation{Ti,Tf}) where {Ti,Tf}
    norm(equation.R)
end

function residual_print(equation::Equation{Ti,Tf}) where {Ti,Tf}
    println("Residual: ", norm(equation.R))
end