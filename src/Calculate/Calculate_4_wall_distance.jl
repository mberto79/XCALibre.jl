export wall_distance
# export residual!

function wall_distance(model, config)
    @info "Calculating wall distance..."

    mesh = model.domain
    y = model.turbulence.y
    (; solvers, schemes, runtime, hardware) = config
    iterations = 500
    
    phi = ScalarField(mesh)
    phif = FaceScalarField(mesh)
    initialise!(phif, 1.0)
    initialise!(phi, 0.0)

    # Assign boundary Conditions         

    phi_eqn = (
        Laplacian{schemes.y.laplacian}(phif, phi) == Source(ConstantScalar(-1.0))
    ) → ScalarEquation(mesh)

    @reset phi_eqn.preconditioner = set_preconditioner(
        solvers.y.preconditioner, phi_eqn, phi.BCs, config)

    @reset phi_eqn.solver = solvers.y.solver(_A(phi_eqn), _b(phi_eqn))

    phiGrad = Grad{schemes.y.gradient}(phi)
    phif = FaceScalarField(mesh)
    grad!(phiGrad, phif, phi, phi.BCs, config)

    n_cells = length(mesh.cells)
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    R_phi = ones(TF, iterations)

    for iteration ∈ 1:iterations
        # @. prev = phi.values
        discretise!(phi_eqn, phi, config)
        apply_boundary_conditions!(phi_eqn, phi.BCs, nothing, config)
        update_preconditioner!(phi_eqn.preconditioner, mesh, config)
        implicit_relaxation!(phi_eqn, phi.values, solvers.y.relax, nothing, config)
        solve_system!(phi_eqn, solvers.y, phi, nothing, config)
        # explicit_relaxation!(phi, prev, solvers.phi.relax, config)
        
        residual!(R_phi, phi_eqn, phi, iteration, nothing, config)
        println("Iteration $iteration: ", R_phi[iteration])

        if R_phi[iteration] < solvers.y.convergence
            @info "Wall distance converged!"
            break
        end
    end
    
    grad!(phiGrad, phif, phi, phi.BCs, config)
    # wallDist = normalDistance(phi, phiGrad)
    # y.values .= wallDist.values
    y.values .= phi.values
    return y.values
    # return phi
end

function normalDistance(phi, phiGrad)
    wallDist = deepcopy(phi)
    @inbounds for i ∈ 1:length(phi.values)
        gradMag = norm(phiGrad.result[i])
        wallDist.values[i] = -gradMag + sqrt(gradMag^2 + 2.0*phi.values[i])
    end
    return wallDist
end