export wall_distance
# export residual!

function wall_distance(phi, model, config)
    @info "Calculating wall distance..."

    mesh = model.domain
    (; solvers, schemes, runtime, hardware) = config
    # (; iterations) = runtime
    iterations = 2000
    
    # phi = ScalarField(mesh)
    y = ScalarField(mesh)
    phif = FaceScalarField(mesh)
    initialise!(phif, 1.0)
    initialise!(phi, 0.0)

    phi_eqn = (
        Laplacian{schemes.phi.laplacian}(phif, phi) == Source(ConstantScalar(-1.0))
    ) → ScalarEquation(mesh)

    @reset phi_eqn.preconditioner = set_preconditioner(
        solvers.phi.preconditioner, phi_eqn, phi.BCs, config)

    @reset phi_eqn.solver = solvers.phi.solver(_A(phi_eqn), _b(phi_eqn))

    phiGrad = Grad{schemes.phi.gradient}(phi)
    phif = FaceScalarField(mesh)
    grad!(phiGrad, phif, phi, phi.BCs, config)

    n_cells = length(mesh.cells)
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    R_phi = ones(TF, iterations)

    for iteration ∈ 1:iterations
        @. prev = phi.values
        # discretise!(phi_eqn, prev, config)
        discretise!(phi_eqn, phi, config)
        apply_boundary_conditions!(phi_eqn, phi.BCs, nothing, config)
        update_preconditioner!(phi_eqn.preconditioner, mesh, config)
        implicit_relaxation!(phi_eqn, phi.values, solvers.phi.relax, nothing, config)
        solve_system!(phi_eqn, solvers.phi, phi, nothing, config)
        # explicit_relaxation!(phi, prev, solvers.phi.relax, config)
        # explicit_relaxation!(p, prev, solvers.p.relax, config)
        # residual!(R_phi, phi_eqn.equation, phi, iteration)
        # grad!(phiGrad, phif, phi, phi.BCs, config)
        
        residual!(R_phi, phi_eqn, phi, iteration, nothing, config)
        println("Iteration $iteration: ", R_phi[iteration])

        if R_phi[iteration] < solvers.phi.convergence
            @info "Wall distance converged!"
            break
        end


        # phi.values .= _A(phi_eqn)\_b(phi_eqn)
        # # discretise!(k_eqn, k, config)
        # # apply_boundary_conditions!(k_eqn, k.BCs, nothing, config)
        # implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
        # # update_preconditioner!(k_eqn.preconditioner, mesh, config)
        # # solve_system!(k_eqn, solvers.k, k, nothing, config)
        # # bound!(k, config)

        # if R_phi[iteration] < solvers.phi.convergence
        #     @info "Wall distance converged!"
        #     break
        # end
    end
    grad!(phiGrad, phif, phi, phi.BCs, config)
    wallDist = normalDistance(phi, phiGrad)
    y.values .= wallDist.values
    return y
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