export run!, set_solver

# (x, stats) = bicgstab(A, b::AbstractVector{T}; c::AbstractVector{T}=b,
# M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
# itmax::Int=0, verbose::Int=0, history::Bool=false)

function run!(
    solver, equation::Equation{Ti,Tv}, phi; 
    atol=1e-8, rtol=1e-7, itmax=500, kwargs...
    ) where {Ti,Tv}

    (; A, b, R, Fx) = equation
    opA = LinearOperator(A)
    R .= b .- mul!(Fx, opA, phi.values)
    solve!(solver, opA, R; itmax=itmax, atol=atol, rtol=rtol, kwargs...)
    phi.values .+= solution(solver)
end

function set_solver(equation::Equation{I,F}, solver) where {I,F}
    solver(equation.A, equation.R)
end