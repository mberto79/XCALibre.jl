export run!, set_solver

# (x, stats) = bicgstab(A, b::AbstractVector{T}; c::AbstractVector{T}=b,
# M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
# itmax::Int=0, verbose::Int=0, history::Bool=false)

function run!(
    solver, equation::Equation{Ti,Tv}, phi; 
    atol=1e-8, rtol=1e-3, itmax=500, kwargs...
    ) where {Ti,Tv}

    (; A, b, R, Fx) = equation
    F = ilu(A, τ = 0.005)
    n = length(b)
    bl = false
    # opM = LinearOperator(Float64, n, n, bl, bl, (y, v) -> forward_substitution!(y, F, v))
    # opN = LinearOperator(Float64, n, n, bl, bl, (y, v) -> backward_substitution!(y, F, v))
    opP = LinearOperator(Float64, n, n, bl, bl, (y, v) -> ldiv!(y, F, v))
    opA = LinearOperator(A)
    Fx .= zero(Tv)
    mul!(Fx, opA, phi.values)
    R .= b .- Fx
    solve!(solver, opA, R; M=opP, itmax=itmax, atol=atol, rtol=rtol, kwargs...)
    phi.values .+= solution(solver)
    nothing
end

function set_solver(equation::Equation{I,F}, solver) where {I,F}
    solver(equation.A, equation.R)
end