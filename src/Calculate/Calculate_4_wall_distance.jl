export wall_distance!
# export residual!

function wall_distance!(model, walls, config; iterations=1000)
    @info "Calculating wall distance..."

    mesh = model.domain
    # (; y, wallBCs) = model.turbulence
    (; y) = model.turbulence
    (; solvers, schemes, runtime, hardware, postprocess) = config

    # set up boundary conditions
    BCs = wall_distance_BCs(mesh, walls, config)
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

    BCs = wallBCs = phi = phi_eqn = phiGrad = phif = prev = nothing
    GC.gc()

    new_config
end

function normal_distance!(y, phi, phiGrad, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phi.values)
    kernel! = _normal_distance!(_setup(backend, workgroup, ndrange)...)
    kernel!(y, phi, phiGrad)
    KernelAbstractions.synchronize(backend)
end

function wall_distance_BCs(mesh, walls, config)
    boundaries_cpu = get_boundaries(mesh.boundaries)
    boundary_names = map(boundary -> boundary.name, boundaries_cpu)
    wall_names = collect(Symbol.(walls))
    missing_walls = setdiff(wall_names, boundary_names)
    isempty(missing_walls) || error("Wall distance patches not found in mesh: $(Tuple(missing_walls)). Available patches: $(Tuple(boundary_names))")

    empty_names = empty_boundary_names(boundaries_cpu, config.boundaries)
    matched_walls = intersect(wall_names, boundary_names)
    isempty(matched_walls) && error("Wall distance needs at least one wall patch")
    warn_omitted_velocity_walls(boundaries_cpu, config.boundaries, matched_walls)

    BCs = []
    for boundary ∈ boundaries_cpu
        boundary_name = boundary.name
        if boundary_name ∈ matched_walls
            push!(BCs, Dirichlet(boundary_name, 0.0))
        elseif boundary_name ∈ empty_names
            push!(BCs, Empty(boundary_name))
        else
            push!(BCs, Extrapolated(boundary_name))
        end
    end
    BCs
end

function empty_boundary_names(boundaries_cpu, boundaries)
    empty_names = Symbol[]
    for field_BCs ∈ boundaries
        for BC ∈ field_BCs
            if typeof(BC) <: Empty
                push!(empty_names, boundaries_cpu[BC.ID].name)
            end
        end
    end
    unique(empty_names)
end

function warn_omitted_velocity_walls(boundaries_cpu, boundaries, wall_names)
    hasproperty(boundaries, :U) || return nothing

    omitted = Symbol[]
    for BC ∈ boundaries.U
        if typeof(BC) <: Union{Wall,RotatingWall}
            name = boundaries_cpu[BC.ID].name
            name ∈ wall_names || push!(omitted, name)
        end
    end

    isempty(omitted) || @warn "Velocity wall patches omitted from wall distance calculation: $(Tuple(unique(omitted))). Add them to turbulence walls if they are physical walls."
    nothing
end

@kernel function _normal_distance!(y, phi, phiGrad)
    i = @index(Global)

    gradMag_raw = norm(phiGrad.result[i])
    gradMag = isfinite(gradMag_raw) ? gradMag_raw : zero(gradMag_raw)
    radicand_raw = gradMag^2 + 2*phi.values[i]
    radicand = isfinite(radicand_raw) ? max(radicand_raw, zero(radicand_raw)) : zero(radicand_raw)
    y.values[i] = max(-gradMag + sqrt(radicand), zero(gradMag))
end
