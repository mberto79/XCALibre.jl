export wall_distance!
# export residual!

function wall_distance!(model, config)
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

    TF = _get_float(mesh)

    phiGrad = Grad{schemes.y.gradient}(phi)
    phif = FaceScalarField(mesh)
    grad!(phiGrad, phif, phi, phi.BCs, zero(TF), config) # assuming time=0

    # n_cells = length(mesh.cells)
    # prev = zeros(TF, n_cells)
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

        if R_phi[iteration] < solvers.y.convergence 
            @info "Wall distance converged in $iteration iterations ($(R_phi[iteration]))"
            break
        elseif iteration == iterations
            @info "Wall distance calculation did not converged ($(R_phi[iteration]))"
        end
    end
    
    grad!(phiGrad, phif, phi, phi.BCs, zero(TF), config) # assuming time=0
    normal_distance!(y, phi, phiGrad, config)
end

function normal_distance!(y, phi, phiGrad, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel! = _normal_distance!(backend, workgroup)
    kernel!(y, phi, phiGrad, ndrange = length(phi.values))
    # KernelAbstractions.synchronize(backend)
end

@kernel function _normal_distance!(y, phi, phiGrad)
    i = @index(Global)

    gradMag = norm(phiGrad.result[i])
    y.values[i] = -gradMag + sqrt(gradMag^2 + 2*phi.values[i])
end