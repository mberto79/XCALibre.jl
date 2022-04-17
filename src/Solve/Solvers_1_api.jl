export run!, set_solver, residual
export update_residual!, update_solution!

function set_solver(equation::Equation{I,F}, solver) where {I,F}
    solver(equation.A, equation.R)
end

function run!(
    solver, equation::Equation{Ti,Tf}, phi; 
    atol=1e-12, rtol=1e-3, itmax=500, kwargs...
    ) where {Ti,Tf}
    (; A) = equation
    F = ilu(A, τ = 0.005)
    
    # Definition of linear operators to reduce allocations during iterations
    opP = LinearOperator(Float64, A.m, A.n, false, false, (y, v) -> ldiv!(y, F, v))
    opA = LinearOperator(A)

    update_residual!(opA, equation, phi)
    # Solving in residual form (allowing to provide an initial guess)
    solve!(solver, opA, equation.R; M=opP, itmax=itmax, atol=atol, rtol=rtol, kwargs...)
    update_solution!(phi, solver; alpha=1.0) # adds solution to initial guess
    update_residual!(opA, equation, phi)
    nothing
end

function update_residual!(opA, equation, phi::ScalarField{Ti,Tf}) where {Ti,Tf}
    (; b, R, Fx) = equation
    Fx .= zero(Tf)
    R .= b .- mul!(Fx, opA, phi.values)
    nothing
end

@inline function update_solution!(phi, solver; alpha=0.3)
    val = phi.values
    sol = solution(solver)
    @inbounds for i ∈ eachindex(val)
        val[i] += sol[i]*alpha # relax solution
    end
end

function residual(equation::Equation{Ti,Tf}) where {Ti,Tf}
    println("Residual: ", norm(equation.R))
end