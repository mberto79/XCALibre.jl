export wall_distance!
# export residual!

function wall_distance!(model, config)
    @info "Calculating wall distance..."

    mesh = model.domain
    (; y, wallBCs) = model.turbulence
    (; solvers, schemes, runtime, hardware, boundaries) = config
    
    phi = ScalarField(mesh)

    phi_eqn = (
        -Laplacian{schemes.y.laplacian}(ConstantScalar(1.0), phi) 
        == 
        Source(ConstantScalar(1.0))
    ) → ScalarEquation(phi, wallBCs.y)

    # @reset phi_eqn.preconditioner = set_preconditioner(
    #     solvers.y.preconditioner, phi_eqn, wallBCs.y, config)

    @reset phi_eqn.preconditioner = set_preconditioner(solvers.y.preconditioner, phi_eqn)

    @reset phi_eqn.solver = _workspace(solvers.y.solver, _b(phi_eqn))

    TF = _get_float(mesh)

    phiGrad = Grad{schemes.y.gradient}(phi)
    phif = FaceScalarField(mesh)
    grad!(phiGrad, phif, phi, wallBCs.y, zero(TF), config) # assuming time=0

    n_cells = length(mesh.cells)
    prev = similar(phi.values)
    # R_phi = ones(TF, iterations)

    iterations = 1000
    for iteration ∈ 1:iterations
        @. prev = phi.values
        discretise!(phi_eqn, phi, config)
        # apply_boundary_conditions!(phi_eqn, wallBCs.y, nothing, 0.0, config) # wrong BCs!!
        apply_boundary_conditions!(phi_eqn, wallBCs.y, nothing, 0.0, config)

        update_preconditioner!(phi_eqn.preconditioner, mesh, config)
        # implicit_relaxation!(phi_eqn, phi.values, solvers.y.relax, nothing, config)
        phi_res = solve_system!(phi_eqn, solvers.y, phi, nothing, config)
        explicit_relaxation!(phi, prev, solvers.y.relax, config)

        if phi_res < solvers.y.convergence 
            @info "Wall distance converged in $iteration iterations ($phi_res)"
            break
        elseif iteration == iterations
            @info "Wall distance calculation did not converged ($phi_res)"
        end
    end
    
    grad!(phiGrad, phif, phi, wallBCs.y, zero(TF), config) # assuming time=0
    normal_distance!(y, phi, phiGrad, config)
    # y.values .= phi.values
end

function normal_distance!(y, phi, phiGrad, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phi.values)
    kernel! = _normal_distance!(_setup(backend, workgroup, ndrange)...)
    kernel!(y, phi, phiGrad)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _normal_distance!(y, phi, phiGrad)
    i = @index(Global)

    gradMag = norm(phiGrad.result[i])
    y.values[i] = (-gradMag + sqrt(gradMag^2 + 2*phi.values[i]))
end