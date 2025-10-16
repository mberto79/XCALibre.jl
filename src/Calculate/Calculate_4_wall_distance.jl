export wall_distance!
# export residual!

function wall_distance!(model, walls, config)
    @info "Calculating wall distance..."

    mesh = model.domain
    # (; y, wallBCs) = model.turbulence
    (; y) = model.turbulence
    (; solvers, schemes, runtime, hardware, postprocess) = config

    # set up boundary conditions
    BCs = []
    boundaries_cpu = get_boundaries(mesh.boundaries)
    boundaries_user = config.boundaries[1] # take first one, a bit frail but should work
    for boundary ∈ boundaries_user
        boundary_name = boundaries_cpu[boundary.ID].name
        if boundary_name ∈ walls
            push!(BCs, Dirichlet(boundary_name, 0.0))
        elseif typeof(boundary) <: Empty
            push!(BCs, Empty(boundary_name))
        else
            push!(BCs, Extrapolated(boundary_name))
        end
    
    end
    wallBCs = assign(
        region=mesh,
        (
            y = [BCs...],
        )
    )

    updated_boundaries = (; config.boundaries..., y = wallBCs.y)
    new_config = Configuration(
        schemes=schemes, solvers=solvers, runtime=runtime, 
        hardware=hardware, postprocess=postprocess, boundaries=updated_boundaries
        )
    (; boundaries) = new_config
    
    phi = ScalarField(mesh)

    phi_eqn = (
        -Laplacian{schemes.y.laplacian}(ConstantScalar(1.0), phi) 
        == 
        Source(ConstantScalar(1.0))
    # ) → ScalarEquation(phi, wallBCs.y) # wallBCs are used when setting up BCs for the user
    ) → ScalarEquation(phi, boundaries.y)

    # @reset phi_eqn.preconditioner = set_preconditioner(
    #     solvers.y.preconditioner, phi_eqn, wallBCs.y, config)

    @reset phi_eqn.preconditioner = set_preconditioner(solvers.y.preconditioner, phi_eqn)

    @reset phi_eqn.solver = _workspace(solvers.y.solver, _b(phi_eqn))

    TF = _get_float(mesh)

    phiGrad = Grad{schemes.y.gradient}(phi)
    phif = FaceScalarField(mesh)
    # grad!(phiGrad, phif, phi, wallBCs.y, zero(TF), config) # assuming time=0
    grad!(phiGrad, phif, phi, boundaries.y, zero(TF), config) # assuming time=0

    n_cells = length(mesh.cells)
    prev = similar(phi.values)
    # R_phi = ones(TF, iterations)

    iterations = 1000
    for iteration ∈ 1:iterations
        @. prev = phi.values
        discretise!(phi_eqn, phi, config)
        # apply_boundary_conditions!(phi_eqn, wallBCs.y, nothing, 0.0, config) # wrong BCs!!
        # apply_boundary_conditions!(phi_eqn, wallBCs.y, nothing, 0.0, config)
        apply_boundary_conditions!(phi_eqn, boundaries.y, nothing, 0.0, config)

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
    
    # grad!(phiGrad, phif, phi, wallBCs.y, zero(TF), config) # assuming time=0
    grad!(phiGrad, phif, phi, boundaries.y, zero(TF), config) # assuming time=0
    normal_distance!(y, phi, phiGrad, config)
    # y.values .= phi.values

    new_config
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