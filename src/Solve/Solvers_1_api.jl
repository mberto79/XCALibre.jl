export SolverSetup
export run!
export residual, residual_print
export relax!, update_residual!, update_solution!
export clear!

struct SolverSetup{S,I,F}
    solver::S 
    iterations::I 
    tolerance::F 
    relax::F 
    itmax::I
    atol::F 
    rtol::F
end
SolverSetup(
    ; solver::S, 
    iterations::I, 
    tolerance::F, 
    relax::F, 
    itmax::I=200, 
    atol::F=1e-12, 
    rtol::F=1e-1
    ) where {S,I,F} = begin
    SolverSetup(solver, iterations, tolerance, relax, itmax, atol, rtol)
end

function run_old!(
    equation::Equation{Ti,Tf}, phiModel, BCs, setup; correct_term=nothing
    ) where {Ti,Tf}
    discretise!(equation, phiModel)
    update_boundaries!(equation, phiModel, BCs)

    (; solver, iterations, tolerance, relax, itmax, atol, rtol) = setup
    (; A, b) = equation
    (; phi) = phiModel.terms.term1
    (; values, mesh) = phi

    solver_alloc = solver(A, b)
    F = ilu(A, τ = 0.005)
    
    # Definition of linear operators to reduce allocations during iterations
    opP = LinearOperator(Float64, A.m, A.n, false, false, (y, v) -> ldiv!(y, F, v))
    opA = LinearOperator(A)

    if correct_term !== nothing 
        bb      = copy(b)
        gradPhi = Grad{Linear}(phi,2)
        phif    = FaceScalarField(mesh)
        gradf   = FaceVectorField(mesh)
    end

    @inbounds for i ∈ 1:iterations
        solve!(
            solver_alloc, opA, b, values; 
            M=opP, itmax=itmax, atol=atol, rtol=rtol
            )
        relax!(phi, solver_alloc, relax)

        if correct_term !== nothing
            nonorthogonal_correction!(gradPhi, gradf, phif, BCs)
            b .= bb
            correct!(equation, correct_term, phif)
        end

        update_residual!(equation, opA, phi)
        if residual(equation) <= tolerance
            residual_print(equation)
            println("Converged in ", i, " iterations")
            break
        end
        residual_print(equation)
    end
    if correct_term !== nothing 
        bb      = nothing
        gradPhi = nothing
        phif    = nothing
        gradf   = nothing
    end
end

function run!(
    equation::Equation{Ti,Tf}, phiModel, BCs, setup; 
    correct_term=nothing, opA, opP
    ) where {Ti,Tf}
    
    equation.b .+= phiModel.sources.source1

    (; solver, iterations, tolerance, relax, itmax, atol, rtol) = setup
    (; A, b, R, Fx) = equation
    (; phi) = phiModel.terms.term1
    (; values, mesh) = phi

    solver_alloc = solver(A, b)

    if correct_term !== nothing 
        bb      = copy(b)
        gradPhi = Grad{Linear}(phi,2)
        phif    = FaceScalarField(mesh)
        gradf   = FaceVectorField(mesh)
    end

    mul!(Fx, opA, values)
    R .= b .- Fx

    @inbounds for i ∈ 1:iterations
        solve!(
            solver_alloc, opA, R; 
            M=opP, itmax=itmax, atol=atol, rtol=rtol
            )
        values .+= relax.*solver_alloc.x

        if correct_term !== nothing
            nonorthogonal_correction!(gradPhi, gradf, phif, BCs)
            b .= bb
            correct!(equation, correct_term, phif)
        end

        mul!(Fx, opA, values)
        R .= b .- Fx
        res = 0.0
        normB = norm(b) 
        normR = norm(R)
        if normB == zero(eltype(b))
            res = 1.0
        else
            res = normR/normB
        end
        if res <= tolerance
            println("Converged in ", i, " iterations. ", "Residual: ", res)
            # println("")
            break
        end
        # println("Residual: ", res)
    end
    if correct_term !== nothing 
        bb      = nothing
        gradPhi = nothing
        phif    = nothing
        gradf   = nothing
    end
end

@inline function relax!(phi, solver, α)
    # phi.values .= (1.0 - α).*phi.values .+ α.*solver.x
    values = phi.values
    x = solver.x
    @inbounds for i ∈ eachindex(values)
        # values[i] = (1.0 - α)*values[i] + α*x[i]
        values[i] = values[i] + α(values[i] - x[i])
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
    norm(equation.R)/norm(equation.b)
end

function residual_print(equation::Equation{Ti,Tf}) where {Ti,Tf}
    println("Residual: ", norm(equation.R)/norm(equation.b))
end

function clear!(phi::ScalarField{I,F}) where {I,F}
    values = phi.values
    zero_type = zero(F)
    @inbounds for i ∈ eachindex(values)
        values[i] = zero_type
    end
end