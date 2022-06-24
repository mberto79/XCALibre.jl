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
    atol::F=1e-15, 
    rtol::F=1e-1
    ) where {S,I,F} = begin
    SolverSetup{S,I,F}(solver, iterations, tolerance, relax, itmax, atol, rtol)
end

function run!(
    equation::Equation{Ti,Tf}, phiModel, BCs, setup; 
    correct_term=nothing, opA, opP, solver_alloc
    ) where {Ti,Tf}
    
    equation.b .+= phiModel.sources.source1

    (; solver, iterations, tolerance, relax, itmax, atol, rtol) = setup
    (; A, b, R, Fx) = equation
    (; phi) = phiModel.terms.term1
    (; values, mesh) = phi

    if correct_term !== nothing 
        bb      = copy(b)
        gradPhi = Grad{Linear}(phi,2)
        phif    = FaceScalarField(mesh)
        gradf   = FaceVectorField(mesh)
    end

    for i ∈ 1:iterations
        solve!(
            # solver_alloc, opA, R; 
            solver_alloc, opA, b, values; 
            M=opP, itmax=itmax, atol=atol, rtol=rtol
            )
        # values .+= relax.*solver_alloc.x
        # @time @. values += relax*(x - values)
        relax!(values, solver_alloc.x, relax)

        if correct_term !== nothing
            nonorthogonal_correction!(gradPhi, gradf, phif, BCs)
            b .= bb
            correct!(equation, correct_term, phif)
        end

        # mul!(Fx, opA, values)
        # # R .= b .- Fx
        # R .= abs.(Fx .- b)
        # for i ∈ eachindex(values)
        #     Fx[i] = abs(A[i,i]*values[i])
        # end
        # R .= R./maximum(Fx)

        # # res = 0.0
        # res = maximum(R)
        # # normB = norm(b) 
        # # normR = norm(R)
        # # if normB == zero(eltype(b))
        # #     res = 1.0
        # # else
        #     # res = normR/normB
        # # end
        # # if res <= tolerance
        #     print(
        #         "Residual: ", res, " (", niterations(solver_alloc), " iterations)\n")
            # break
        # end
        # println("Residual: ", res)
    end
    if correct_term !== nothing 
        bb      = nothing
        gradPhi = nothing
        phif    = nothing
        gradf   = nothing
    end
end

@inline function relax!(phi, phi_new, α)
    @inbounds for i ∈ eachindex(phi)
        phi[i] += α*(phi_new[i] - phi[i])
        # phi[i] = α*phi_new[i] + (1.0 - α)*phi[i]
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