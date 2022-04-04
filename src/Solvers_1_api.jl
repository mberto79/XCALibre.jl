export solver!, solver

# (x, stats) = bicgstab(A, b::AbstractVector{T}; c::AbstractVector{T}=b,
# M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
# itmax::Int=0, verbose::Int=0, history::Bool=false)

function solver!(solver, equation, phi; iterations=100, kwargs...)
    (; A, b, R, Fx) = equation
    (; values) = phi
    F = ilu(A, τ = 0.05)
    n = length(b)
    bl = false
    opM = LinearOperator(Float64, n, n, bl, bl, (y, v) -> forward_substitution!(y, F, v))
    opN = LinearOperator(Float64, n, n, bl, bl, (y, v) -> backward_substitution!(y, F, v))
    # opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, F, v))
    for iteration ∈ 1:iterations
        R .= b .- mul!(Fx, A, values)
        solve!(solver, A, R; itmax=1, M=opM, N=opN, kwargs...)
        values .+= solution(solver)
    end
    # solve!(solver, A, b; itmax=iterations, kwargs...)
    # values .= solution(solver)
end

function solver(equation::Equation{I,F}) where {I,F}
    BicgstabSolver(equation.A, equation.R)
    # BicgstabSolver(equation.A, equation.b)
end