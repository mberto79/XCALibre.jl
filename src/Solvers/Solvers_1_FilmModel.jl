export filmModel!

function filmModel!(
    model, config;
    output=VTK(),#, pref=nothing, ncorrectors=
    inner_loops=2
)
    #print("Using film model\n")
    residuals = setup_FilmModel_Solver(
        FilmModel, model, config,
        output=output,
        inner_loops=inner_loops
    )
    
    return residuals
end

function setup_FilmModel_Solver(solver_variant, model, config;
    output=VTK(), inner_loops=2)

    (; solvers, schemes, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, h) = model.momentum
    mesh = model.domain
    

    @info "Pre-allocating fields"
    phif_U = FaceScalarField(mesh)
    filmVelocityFlux = FaceScalarField(mesh)
    nu_h = ScalarField(mesh)
    Sm = ScalarField(mesh)
    surfaceFluxDiv = ScalarField(mesh)
    initialise!(Sm, 0)
    h∇PL = VectorField(mesh)
    Ph = VectorField(mesh)
    τθw = VectorField(mesh)
    Df = FaceScalarField(mesh)
    

    @info "Defining models"
    h_divergence = schemes.h.divergence === Linear ? LUST : schemes.h.divergence
    if schemes.h.divergence === Linear
        @info "Using LUST for film h convection; pure Linear h convection is not checkerboard-safe for this solver"
    end
    if h_divergence === BoundedUpwind
        error("BoundedUpwind is not supported for film h convection because the final conservative film flux cannot yet be reconstructed consistently for that scheme.")
    end

    # Edit
    @info "U equation still need updating"
    U_eqn = (
        Time{schemes.U.time}(h, U)
        + Divergence{schemes.U.divergence}(phif_U,U)
        + Si(nu_h, U)
        ==
        - Source(h∇PL)
        + Source(Ph)
        + Source(τθw)
        
    ) → VectorEquation(U, boundaries.U)

    h_eqn = (
        Time{schemes.h.time}(h)
        + Divergence{h_divergence}(filmVelocityFlux,h)
        - Laplacian{schemes.h.laplacian}(Df,h)
        ==
        - Source(surfaceFluxDiv)
        + Source(Sm)
    ) → ScalarEquation(h, boundaries.h)

    @info "Initialising preconditioners"

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset h_eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, h_eqn)

    @info "Pre-allocating solvers"

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset h_eqn.solver = _workspace(solvers.h.solver, _b(h_eqn))

    @info "Initialising turbulence model"
    #p_eqn = (Time{schemes.h.time}(rho_l,h)==Source(Sm)) → ScalarEquation(h, boundaries.h)
    #turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals = solver_variant(
        model, #turbulenceModel,
         U_eqn, h_eqn, config; output=output, inner_loops=inner_loops
    )
end

function FilmModel(
    model, #turbulenceModel,
     U_eqn, h_eqn, config;
    output=VTK(), ncorrectors=0, inner_loops=2
)
    (; U, h, Uf, hf, coeffs) = model.momentum
    (; rho, nu) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, boundaries, postprocess) = config
    (; iterations, write_interval, dt) = runtime
    backend = config.hardware.backend
    
    TF = _get_float(mesh)
    n_cells = length(mesh.cells)
    capillaryDtFaces = KernelAbstractions.zeros(backend, TF, length(mesh.faces))

    dt_cpu = zeros(TF, 1)

    postprocess = convert_time_to_iterations(postprocess, model, dt, iterations)
    phif_U = get_flux(U_eqn, 2)
    phif = FaceScalarField(mesh)
    # Parabolic velocity profile correction (Meredith Eq 2 Dv): ∫u² dy = (6/5) h ū²
    profile_factor = TF(6) / TF(5)
    nu_h = get_flux(U_eqn, 3)

    h∇PL = get_source(U_eqn, 1)
    Ph = get_source(U_eqn,2)
    τθw = get_source(U_eqn,3)

    filmVelocityFlux = get_flux(h_eqn, 2)
    Df = get_flux(h_eqn, 3)
    surfaceFluxDiv = get_source(h_eqn, 1)

    outputWriter = initialise_writer(output, model.domain)

    @info "Allocating working memory"

    G = coeff_vector(coeffs, :gravity, SVector{3,TF}(0, 0, -9.81), TF)

    # Define aux fields
    mdotf = FaceScalarField(mesh)
    filmSurfaceFlux = FaceScalarField(mesh)
    PLf = FaceScalarField(mesh)
    ∇PL = Grad{Gauss}(PLf)
    P_hydrf = FaceScalarField(mesh)
    P_surf = ScalarField(mesh)
    P_surff = FaceScalarField(mesh)
    ∇P_hydr = Grad{Gauss}(P_hydrf)
    ∇P_surf = Grad{Gauss}(P_surff)

    tempU = VectorField(mesh)
    Δh = ScalarField(mesh)
    Δhf = FaceScalarField(mesh)
    ∇hf = FaceVectorField(mesh)

    w = ScalarField(mesh)
    wf = FaceScalarField(mesh)
    wcf = FaceScalarField(mesh)
    ∇w = Grad{schemes.h.gradient}(w)

    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)
    surfaceNormal = VectorField(mesh)
    gravityTangent = VectorField(mesh)
    gNormalf = FaceScalarField(mesh)
    
    initialise_film_geometry!(surfaceNormal, gravityTangent, gNormalf, G, config)

    mesh_boundaries = get_boundaries(mesh.boundaries)
    wetting_BCs = [
        hBC isa AbstractDirichlet ? Dirichlet(boundary.name, 1) : Zerogradient(boundary.name)
        for (hBC, boundary) in zip(boundaries.h, mesh_boundaries)
    ]
    internal_BCs = assign(
        region=mesh,
        (
            w = wetting_BCs,
        )
    )

    # Pre-allocate auxiliary variables
    wetting_mode = if hasproperty(coeffs, :wetting_mode)
        efm_wetting_mode(coeffs.wetting_mode)
    else
        efm_wetting_mode()
    end
    h_old = KernelAbstractions.zeros(backend, TF, n_cells)
    prev = KernelAbstractions.zeros(backend, TF, n_cells)

    # Pre-allocate vectors to hold residuals
    R_u = zeros(TF, iterations)
    R_h = zeros(TF, iterations)
    courant = zeros(TF, iterations)
    cellsCourant = KernelAbstractions.zeros(backend, TF, n_cells)
    cellsFilmCourant = KernelAbstractions.zeros(backend, TF, n_cells)

    # Initial calculations
    time = zero(TF) # assuming time = 0

    bound_h_nonnegative!(h, config)

    update_wetting_fields!(w, wf, h, internal_BCs.w, coeffs.h_crit, wetting_mode, time, config)
    update_capillary_face_wetting!(wcf, w, wf, config)
    
    #@info "need to readd Pg term - Coupling term for other phase"
    Pg = zero(TF) # Test Pg term set to zero, as the gradient is found this value doesn't matter
    update_film_pressure_fields!(
        Δh, ∇hf, hf, Δhf, P_hydrf, P_surf, P_surff, h,
        boundaries.h, rho.values[1],
        coeffs.σ, gNormalf, time, config
    )

    update_film_fluxes!(
        phif, filmVelocityFlux, mdotf, Uf, U, hf, w, wf,
        boundaries.U, boundaries.h, time, config, phif_U, profile_factor
    )
    update_capillary_dt!(
        config.runtime, capillaryDtFaces, mesh, hf, wf, rho.values[1], coeffs, config
    )
    copyto!(dt_cpu, config.runtime.dt)

    update_film_viscous_source!(nu_h, nu, h, coeffs.h_floor, config)
    update_liquid_pressure!(PLf, P_hydrf, P_surff, Pg, config)

    grad!(∇PL, PLf, config)

    grad!(∇w, wf, w, internal_BCs.w, time, config)

    update_film_force_sources!(
        Ph, h∇PL, τθw, h, ∇PL, ∇w, gravityTangent,
        coeffs, rho.values[1], config
    )

    @info "Starting loops"
    
    progress = Progress(iterations; dt=1.0, showspeed=true)
            
    xdir, ydir, zdir = XDir(), YDir(), ZDir()
    #rh = 0
    rx = ry = rz = zero(TF)
    @time for iteration ∈ 1:iterations
        min_capillary_dt = update_capillary_dt!(
            config.runtime, capillaryDtFaces, mesh, hf, wf, rho.values[1], coeffs, config
        )
        copyto!(dt_cpu, config.runtime.dt)
        time += dt_cpu[1]
        step_dt = dt_cpu[1]

        copyto!(h_old, h.values) # store previous h before inner loop

        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config, time=time)
        project_vector_field_to_surface!(U, surfaceNormal, config)
        
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_film_pressure_source!(U_eqn, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rho.values[1], h, surfaceNormal, config)

        H!(Hv, U, U_eqn, config)
        correct_film_velocity!(
            tempU, Hv, h, P_hydrf, P_surff, rD, rho.values[1],
            surfaceNormal, config; include_hydrostatic=false, include_surface=false
        )
        rh = zero(TF)
        for i ∈ 1:inner_loops
            H!(Hv, tempU, U_eqn, config)

            correct_film_velocity!(
                tempU, Hv, h, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rD, rho.values[1],
                surfaceNormal, config; include_hydrostatic=false, include_surface=false
            )
            update_film_fluxes!(
                phif, filmVelocityFlux, mdotf, Uf, tempU, hf, w, wf,
                boundaries.U, boundaries.h, time, config, phif_U, profile_factor
            )

            getDf!(Df, rDf, hf, wcf, gNormalf, config)
            update_film_surface_flux!(
                filmSurfaceFlux, rDf, hf, wf, w, filmVelocityFlux, boundaries.h,
                P_surf, P_surff, rho.values[1], config
            )
            div!(surfaceFluxDiv, filmSurfaceFlux, config)
            
            copyto!(prev, h.values)
            rh = solve_film_h_equation!(h_eqn, h, h_old, boundaries.h, solvers.h, config, time=time)

            if i == inner_loops
                explicit_relaxation!(h, prev, 1.0, config)
            else
                explicit_relaxation!(h, prev, solvers.h.relax, config)
            end 

            bound_h_nonnegative!(h, config)
            update_wetting_fields!(w, wf, h, internal_BCs.w, coeffs.h_crit, wetting_mode, time, config)
            update_capillary_face_wetting!(wcf, w, wf, config)

            update_film_pressure_fields!(
                Δh, ∇hf, hf, Δhf, P_hydrf, P_surf, P_surff, h,
                boundaries.h, rho.values[1],
                coeffs.σ, gNormalf, time, config
            )
            getDf!(Df, rDf, hf, wcf, gNormalf, config)

            correct_film_velocity!(
                U, Hv, h, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rD, rho.values[1],
                surfaceNormal, config; include_hydrostatic=true
            )
            correct_film_velocity!(
                tempU, Hv, h, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rD, rho.values[1],
                surfaceNormal, config; include_hydrostatic=false, include_surface=false
            )
            update_film_fluxes!(
                phif, filmVelocityFlux, mdotf, Uf, tempU, hf, w, wf,
                boundaries.U, boundaries.h, time, config, phif_U, profile_factor
            )
            update_film_surface_flux!(
                filmSurfaceFlux, rDf, hf, wf, w, filmVelocityFlux, boundaries.h,
                P_surf, P_surff, rho.values[1], config
            )
            correct_film_flux2!(phif, filmSurfaceFlux, Df, h_eqn, w, wf, boundaries.h, time, config)
        end

        update_film_viscous_source!(nu_h, nu, h, coeffs.h_floor, config)
        update_liquid_pressure!(PLf, P_hydrf, P_surff, Pg, config)

        grad!(∇PL, PLf, config)

        update_wetting_fields!(w, wf, h, internal_BCs.w, coeffs.h_crit, wetting_mode, time, config)
        update_capillary_face_wetting!(wcf, w, wf, config)

        apply_wetting_to_velocity!(U, w, config)

        correct_film_velocity!(
            tempU, Hv, h, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rD, rho.values[1],
            surfaceNormal, config; include_hydrostatic=false, include_surface=false
        )
        update_film_fluxes!(
            phif, filmVelocityFlux, mdotf, Uf, tempU, hf, w, wf,
            boundaries.U, boundaries.h, time, config
        )
        apply_film_convection_flux!(phif, h_eqn.model.terms[2], h, config)
        scale_face_flux!(phif_U, phif, profile_factor, config)

        grad!(∇w, wf, w, internal_BCs.w, time, config)
        update_film_force_sources!(
            Ph, h∇PL, τθw, h, ∇PL, ∇w, gravityTangent,
            coeffs, rho.values[1], config
        )

        R_u[iteration] = max(rx, ry, rz)
        R_h[iteration] = rh

        maxCourant = max_courant_number!(cellsCourant, model, config)
        maxFilmCourant = max_film_courant_number!(cellsFilmCourant, phif, hf, coeffs.h_crit, config)
        limitingCourant = max(maxCourant, maxFilmCourant)
        courant[iteration] = limitingCourant
        update_dt!(config.runtime, limitingCourant)
        update_capillary_dt!(
            config.runtime, capillaryDtFaces, mesh, hf, wf, rho.values[1], coeffs, config
        )

        ProgressMeter.next!(
            progress, showvalues = [
                (:time, time),
                (:dt, step_dt),
                (:Co, limitingCourant),
                (:U, R_u[iteration]),
                (:h, R_h[iteration]),
                #turbulenceModel.state.residuals...
            ]
        )

        #runtime_postprocessing!(postprocess, iteration, iterations)

        if iteration % write_interval + signbit(write_interval) == 0

            save_output_film(model, outputWriter, iteration, time, config, w)
            save_postprocessing(postprocess, iteration, time, mesh, outputWriter, config.boundaries)
        end

    end # end for loop

    return (U=R_u, h=R_h, courant=courant)
end

function coeff_vector(coeffs, name::Symbol, default::SVector{3,TF}, ::Type{TF}) where TF
    value = hasproperty(coeffs, name) ? getproperty(coeffs, name) : default
    return SVector{3,TF}(value)
end

function initialise_film_geometry!(surfaceNormal, gravityTangent, gNormalf, G, config)
    (; mesh) = surfaceNormal
    (; hardware) = config
    (; backend, workgroup) = hardware

    cell_kernel! = _initialise_film_cell_geometry!(_setup(backend, workgroup, length(mesh.cells))...)
    cell_kernel!(surfaceNormal, gravityTangent, mesh, G)

    face_kernel! = _initialise_film_face_geometry!(_setup(backend, workgroup, length(mesh.faces))...)
    face_kernel!(gNormalf, surfaceNormal, mesh, G, length(mesh.boundary_cellsID))
end

@kernel function _initialise_film_cell_geometry!(surfaceNormal, gravityTangent, mesh, G)
    cID = @index(Global)

    @uniform begin
        cells = mesh.cells
        cell_nodes = mesh.cell_nodes
        nodes = mesh.nodes
    end

    @inbounds begin
        nrange = cells[cID].nodes_range
        n = zero(G)

        if length(nrange) >= 3
            prev = cell_nodes[last(nrange)]
            for ni in nrange
                curr = cell_nodes[ni]
                p = nodes[prev].coords
                q = nodes[curr].coords
                n += SVector(
                    (p[2] - q[2]) * (p[3] + q[3]),
                    (p[3] - q[3]) * (p[1] + q[1]),
                    (p[1] - q[1]) * (p[2] + q[2])
                )
                prev = curr
            end
        end

        nmag = sqrt(n ⋅ n)
        if nmag <= eps(one(nmag))
            n = SVector(zero(nmag), zero(nmag), one(nmag))
            nmag = one(nmag)
        end
        n /= nmag

        if G ⋅ n > zero(nmag)
            n = -n
        end

        t = G - (G ⋅ n)*n
        tmag = sqrt(t ⋅ t)
        if tmag <= eps(one(tmag))
            t = zero(G)
        end

        surfaceNormal[cID] = n
        gravityTangent[cID] = t
    end
end

@kernel function _initialise_film_face_geometry!(gNormalf, surfaceNormal, mesh, G, n_bfaces)
    fID = @index(Global)

    @uniform begin
        faces = mesh.faces
        boundary_cellsID = mesh.boundary_cellsID
    end

    @inbounds begin
        ownerCells = faces[fID].ownerCells
        c1 = ifelse(fID <= n_bfaces, boundary_cellsID[fID], ownerCells[1])
        c2 = ownerCells[2]
        n = surfaceNormal[c1]
        if fID > n_bfaces
            n2 = surfaceNormal[c2]
            n += ifelse(n ⋅ n2 < zero(n[1]), -n2, n2)
            nmag = sqrt(n ⋅ n)
            if nmag > eps(nmag)
                n /= nmag
            end
        end
        if G ⋅ n > zero(n[1])
            n = -n
        end
        gNormalf[fID] = G ⋅ n
    end
end

function update_film_pressure_fields!(
    Δh, ∇hf, hf, Δhf, P_hydrf, P_surf, P_surff, h,
    hBCs, rho, σ, gNormalf, time, config
)
    surface_gradient!(∇hf, hf, h, hBCs, time, config)
    div!(Δh, ∇hf, config)
    interpolate!(Δhf, Δh, config)

    (; hardware) = config
    (; backend, workgroup) = hardware

    cell_kernel! = _update_film_pressure_cells!(_setup(backend, workgroup, length(h))...)
    cell_kernel!(P_surf, Δh, σ)

    face_kernel! = _update_film_pressure_faces!(_setup(backend, workgroup, length(hf))...)
    face_kernel!(P_hydrf, P_surff, hf, Δhf, rho, σ, gNormalf)
end

@kernel function _update_film_pressure_cells!(P_surf, Δh, σ)
    i = @index(Global)

    @inbounds begin
        P_surf[i] = σ * Δh[i]
    end
end

@kernel function _update_film_pressure_faces!(P_hydrf, P_surff, hf, Δhf, rho, σ, gNormalf)
    i = @index(Global)

    @inbounds begin
        P_hydrf[i] = rho * hf[i] * gNormalf[i]
        P_surff[i] = σ * Δhf[i]
    end
end

function update_liquid_pressure!(PLf, P_hydrf, P_surff, Pg, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(PLf)
    kernel! = _update_liquid_pressure!(_setup(backend, workgroup, ndrange)...)
    kernel!(PLf, P_hydrf, P_surff, Pg)
end

@kernel function _update_liquid_pressure!(PLf, P_hydrf, P_surff, Pg)
    i = @index(Global)

    @inbounds begin
        PLf[i] = Pg - P_hydrf[i] - P_surff[i]
    end
end

function update_film_fluxes!(
    phif, filmVelocityFlux, mdotf, Uf, U, hf, w, wf, UBCs, hBCs, time, config
)
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, UBCs, time, config)
    flux!(mdotf, Uf, config)
    apply_film_boundary_flux_policy!(mdotf, hBCs, config)
    apply_wetted_velocity_flux!(filmVelocityFlux, mdotf, w, wf, config)
    apply_film_flux!(phif, filmVelocityFlux, hf, config)
end

function update_film_fluxes!(
    phif, filmVelocityFlux, mdotf, Uf, U, hf, w, wf, UBCs, hBCs, time, config,
    scaledFlux, scale
)
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, UBCs, time, config)
    flux!(mdotf, Uf, config)
    apply_film_boundary_flux_policy!(mdotf, hBCs, config)
    apply_wetted_velocity_flux!(filmVelocityFlux, mdotf, w, wf, config)
    apply_film_flux!(phif, scaledFlux, filmVelocityFlux, hf, scale, config)
end

function scale_face_flux!(scaledFlux, flux, scale, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(scaledFlux)
    kernel! = _scale_face_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(scaledFlux, flux, scale)
end

@kernel function _scale_face_flux!(scaledFlux, flux, scale)
    i = @index(Global)

    @inbounds begin
        scaledFlux[i] = scale * flux[i]
    end
end

function update_film_viscous_source!(nu_h, nu, h, h_floor, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(nu_h)
    kernel! = _update_film_viscous_source!(_setup(backend, workgroup, ndrange)...)
    kernel!(nu_h, nu, h, h_floor)
end

@kernel function _update_film_viscous_source!(nu_h, nu, h, h_floor)
    i = @index(Global)

    @inbounds begin
        nu_h[i] = 3 * nu[i] / max(h[i], h_floor)
    end
end

function apply_wetting_to_velocity!(U, w, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(U)
    kernel! = _apply_wetting_to_velocity!(_setup(backend, workgroup, ndrange)...)
    kernel!(U, w)
end

@kernel function _apply_wetting_to_velocity!(U, w)
    i = @index(Global)

    @inbounds begin
        U[i] = U[i] * w[i]
    end
end

function update_film_force_sources!(
    Ph, h∇PL, τθw, h, ∇PL, ∇w, gravityTangent, coeffs, rho, config
)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(h)
    kernel! = _update_film_force_sources!(_setup(backend, workgroup, ndrange)...)
    contact_line_scale = coeffs.β * coeffs.σ / rho * (1 - cosd(coeffs.θm))
    kernel!(
        Ph, h∇PL, τθw, h, ∇PL, ∇w, gravityTangent,
        contact_line_scale, rho, coeffs.h_crit
    )
end

@kernel function _update_film_force_sources!(
    Ph, h∇PL, τθw, h, ∇PL, ∇w, gravityTangent, contact_line_scale, rho, h_crit
)
    i = @index(Global)

    @inbounds begin
        hi = h[i]
        grad_w = ∇w.result[i]
        film_weight = hi / (hi + h_crit)

        Ph[i] = hi * gravityTangent[i]
        h∇PL[i] = hi * ∇PL.result[i] / rho
        τθw[i] = film_weight * contact_line_scale * grad_w
    end
end

function update_wetting_fields!(w, wf, h, wBCs, h_crit, time, config)
    update_wetting_fields!(w, wf, h, wBCs, h_crit, efm_wetting_mode(), time, config)
end

function update_wetting_fields!(w, wf, h, wBCs, h_crit, wetting_mode, time, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    wetting_mode = efm_wetting_mode(wetting_mode)
    ndrange = length(w)
    if wetting_mode === Val(:allwet)
        kernel! = _set_wetting_field!(_setup(backend, workgroup, ndrange)...)
        kernel!(w, one(h_crit))
    elseif wetting_mode === Val(:smooth)
        kernel! = _update_smooth_wetting_field!(_setup(backend, workgroup, ndrange)...)
        kernel!(w, h, h_crit)
    else
        kernel! = _update_wetting_field!(_setup(backend, workgroup, ndrange)...)
        kernel!(w, h, h_crit)
    end

    interpolate!(wf, w, config)
    correct_boundaries!(wf, w, wBCs, time, config)
    clamp_face_wetting!(wf, config)
end

function efm_wetting_mode(::Val{mode}) where mode
    efm_wetting_mode(mode)
end

function efm_wetting_mode(mode::Symbol)
    efm_wetting_mode(String(mode))
end

function efm_wetting_mode(mode::AbstractString="smooth")
    normalized = lowercase(strip(mode))

    if normalized == "hard"
        return Val(:hard)
    elseif normalized == "smooth" || normalized == "smoothed"
        return Val(:smooth)
    elseif normalized == "allwet"
        return Val(:allwet)
    end

    throw(ArgumentError("wetting_mode must be \"hard\", \"smooth\", \"smoothed\", or \"allwet\"; got $(repr(mode))"))
end

@kernel function _set_wetting_field!(w, value)
    i = @index(Global)

    @inbounds begin
        w[i] = value
    end
end

@kernel function _update_wetting_field!(w, h, h_crit)
    i = @index(Global)

    @inbounds begin
        w[i] = ifelse(h[i] > h_crit, one(h[i]), zero(h[i]))
    end
end

@inline function smooth_wetting_value(h, h_crit)
    hcrit = max(h_crit, eps(h_crit))
    s = clamp(h / hcrit, zero(h), one(h))
    return s * s * (3 - 2 * s)
end

@kernel function _update_smooth_wetting_field!(w, h, h_crit)
    i = @index(Global)

    @inbounds begin
        w[i] = smooth_wetting_value(h[i], h_crit)
    end
end

function clamp_face_wetting!(wf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(wf)
    kernel! = _clamp_face_wetting!(_setup(backend, workgroup, ndrange)...)
    kernel!(wf)
end

@kernel function _clamp_face_wetting!(wf)
    i = @index(Global)

    @inbounds begin
        wf[i] = clamp(wf[i], zero(wf[i]), one(wf[i]))
    end
end

function update_capillary_face_wetting!(wcf, w, wf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(wcf)
    kernel! = _update_capillary_face_wetting!(_setup(backend, workgroup, ndrange)...)
    kernel!(wcf, w, wf, length(w.mesh.boundary_cellsID))
end

@kernel function _update_capillary_face_wetting!(wcf, w, wf, n_bfaces)
    fID = @index(Global)

    @uniform begin
        faces = wcf.mesh.faces
    end

    @inbounds begin
        if fID <= n_bfaces
            wcf[fID] = wf[fID]
        else
            ownerCells = faces[fID].ownerCells
            wcf[fID] = min(w[ownerCells[1]], w[ownerCells[2]])
        end
    end
end

function apply_film_boundary_flux_policy!(faceFlux, hBCs::Tuple, config)
    for hBC in hBCs
        apply_film_boundary_flux_policy!(faceFlux, hBC, config)
    end
end

apply_film_boundary_flux_policy!(faceFlux, hBC::AbstractDirichlet, config) = nothing

function apply_film_boundary_flux_policy!(faceFlux, hBC, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(hBC.IDs_range)
    kernel! = _apply_film_outflow_boundary_flux_policy!(_setup(backend, workgroup, ndrange)...)
    kernel!(faceFlux, hBC.IDs_range)
end

@kernel function _apply_film_outflow_boundary_flux_policy!(faceFlux, IDs_range)
    i = @index(Global)
    fID = IDs_range[i]

    @inbounds begin
        faceFlux[fID] = max(faceFlux[fID], zero(faceFlux[fID]))
    end
end

function apply_wetted_velocity_flux!(filmVelocityFlux, mdotf, w, wf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(filmVelocityFlux)
    kernel! = _apply_wetted_velocity_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(filmVelocityFlux, mdotf, w, wf, length(w.mesh.boundary_cellsID))
end

@kernel function _apply_wetted_velocity_flux!(filmVelocityFlux, mdotf, w, wf, n_bfaces)
    fID = @index(Global)

    @uniform begin
        faces = filmVelocityFlux.mesh.faces
    end

    @inbounds begin
        flux = mdotf[fID]
        wetting = if fID <= n_bfaces
            wf[fID]
        else
            ownerCells = faces[fID].ownerCells
            donor = ifelse(flux >= zero(flux), ownerCells[1], ownerCells[2])
            w[donor]
        end
        filmVelocityFlux[fID] = flux * clamp(wetting, zero(wetting), one(wetting))
    end
end

function apply_film_flux!(phif, filmVelocityFlux, hf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phif)
    kernel! = _apply_film_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, filmVelocityFlux, hf)
end

function apply_film_flux!(phif, scaledFlux, filmVelocityFlux, hf, scale, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phif)
    kernel! = _apply_film_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, scaledFlux, filmVelocityFlux, hf, scale)
end

@kernel function _apply_film_flux!(phif, filmVelocityFlux, hf)
    fID = @index(Global)

    @inbounds begin
        phif[fID] = filmVelocityFlux[fID] * hf[fID]
    end
end

@kernel function _apply_film_flux!(phif, scaledFlux, filmVelocityFlux, hf, scale)
    fID = @index(Global)

    @inbounds begin
        flux = filmVelocityFlux[fID] * hf[fID]
        phif[fID] = flux
        scaledFlux[fID] = scale * flux
    end
end

function solve_film_h_equation!(
    h_eqn, h, h_old, hBCs, solversetup, config; time=nothing
)
    discretise!(h_eqn, h_old, config)
    apply_boundary_conditions!(h_eqn, hBCs, nothing, time, config)
    setReference!(h_eqn, nothing, 1, config)
    update_preconditioner!(h_eqn.preconditioner, h.mesh, config)
    solve_system!(h_eqn, solversetup, h, nothing, config)
end


function save_output_film(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config, w
    ) where {T,F,SO,M,Tu,E,D,BI}
    args = (
            ("U", model.momentum.U), 
            ("h", model.momentum.h),
            ("w", w)
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

function getDf!(Df, rDf, hf, wf, gNormalf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    ndrange = length(hf)
    kernel! = _getDf!(_setup(backend, workgroup, ndrange)...)
    kernel!(Df, rDf, hf, wf, gNormalf)
end

@kernel function _getDf!(Df, rDf, hf, wf, gNormalf)
    i = @index(Global)

    @inbounds begin
        Df[i] = -rDf[i] * hf[i]^2 * wf[i] * gNormalf[i]
    end
end

function correct_film_surface_flux!(phif, rDf, hf, wf, P_surf, P_surff, rho, config)
    (; mesh) = phif
    (; faces, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    n_bfaces = length(boundary_cellsID)
    n_ifaces = length(faces) - n_bfaces

    if n_ifaces > 0
        kernel! = _correct_film_surface_flux!(_setup(backend, workgroup, n_ifaces)...)
        kernel!(phif, rDf, hf, wf, P_surf, faces, rho, n_bfaces)
    end

    if n_bfaces > 0
        kernel! = _correct_film_surface_flux_boundary!(_setup(backend, workgroup, n_bfaces)...)
        kernel!(phif, rDf, hf, wf, P_surf, P_surff, faces, boundary_cellsID, rho)
    end
end

function update_film_surface_flux!(
    filmSurfaceFlux, rDf, hf, wf, w, filmVelocityFlux, hBCs,
    P_surf, P_surff, rho, config
)
    correct_film_surface_flux!(filmSurfaceFlux, rDf, hf, wf, P_surf, P_surff, rho, config)
    apply_donor_wetting_to_flux!(filmSurfaceFlux, w, wf, config)
    limit_dirichlet_correction_flux!(filmSurfaceFlux, filmVelocityFlux, hBCs, config)
end

function zero_face_flux!(faceFlux, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(faceFlux)
    kernel! = _zero_face_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(faceFlux)
end

function limit_dirichlet_correction_flux!(correctionFlux, referenceFlux, hBCs::Tuple, config)
    for hBC in hBCs
        limit_dirichlet_correction_flux!(correctionFlux, referenceFlux, hBC, config)
    end
end

limit_dirichlet_correction_flux!(correctionFlux, referenceFlux, hBC, config) = nothing

function limit_dirichlet_correction_flux!(correctionFlux, referenceFlux, hBC::AbstractDirichlet, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(hBC.IDs_range)
    kernel! = _limit_dirichlet_correction_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(correctionFlux, referenceFlux, hBC.IDs_range)
end

@kernel function _limit_dirichlet_correction_flux!(correctionFlux, referenceFlux, IDs_range)
    i = @index(Global)
    fID = IDs_range[i]

    @inbounds begin
        ref = referenceFlux[fID]
        corr = correctionFlux[fID]
        correctionFlux[fID] = ifelse(
            ref < zero(ref),
            min(corr, -ref),
            ifelse(ref > zero(ref), max(corr, -ref), zero(corr))
        )
    end
end

function limit_dirichlet_total_flux!(totalFlux, referenceFlux, hBCs::Tuple, config)
    for hBC in hBCs
        limit_dirichlet_total_flux!(totalFlux, referenceFlux, hBC, config)
    end
end

limit_dirichlet_total_flux!(totalFlux, referenceFlux, hBC, config) = nothing

function limit_dirichlet_total_flux!(totalFlux, referenceFlux, hBC::AbstractDirichlet, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(hBC.IDs_range)
    kernel! = _limit_dirichlet_total_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(totalFlux, referenceFlux, hBC.IDs_range)
end

@kernel function _limit_dirichlet_total_flux!(totalFlux, referenceFlux, IDs_range)
    i = @index(Global)
    fID = IDs_range[i]

    @inbounds begin
        ref = referenceFlux[fID]
        flux = totalFlux[fID]
        totalFlux[fID] = ifelse(
            ref < zero(ref),
            min(flux, zero(flux)),
            ifelse(ref > zero(ref), max(flux, zero(flux)), zero(flux))
        )
    end
end

@kernel function _zero_face_flux!(faceFlux)
    fID = @index(Global)

    @inbounds begin
        faceFlux[fID] = zero(faceFlux[fID])
    end
end

function apply_donor_wetting_to_flux!(faceFlux, w, wf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(faceFlux)
    kernel! = _apply_donor_wetting_to_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(faceFlux, w, wf, length(w.mesh.boundary_cellsID))
end

@kernel function _apply_donor_wetting_to_flux!(faceFlux, w, wf, n_bfaces)
    fID = @index(Global)

    @uniform begin
        faces = faceFlux.mesh.faces
    end

    @inbounds begin
        flux = faceFlux[fID]
        wetting = if fID <= n_bfaces
            wf[fID]
        else
            ownerCells = faces[fID].ownerCells
            donor = ifelse(flux >= zero(flux), ownerCells[1], ownerCells[2])
            w[donor]
        end
        faceFlux[fID] = flux * clamp(wetting, zero(wetting), one(wetting))
    end
end

@kernel function _correct_film_surface_flux!(phif, rDf, hf, wf, P_surf, faces, rho, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin
        (; ownerCells, delta, area) = faces[fID]
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        snGradP = (P_surf[cID2] - P_surf[cID1]) / delta
        hfi = hf[fID]
        phif[fID] = rDf[fID] * hfi * hfi * wf[fID] * area * snGradP / rho
    end
end

@kernel function _correct_film_surface_flux_boundary!(
    phif, rDf, hf, wf, P_surf, P_surff, faces, boundary_cellsID, rho
)
    fID = @index(Global)

    @inbounds begin
        cID = boundary_cellsID[fID]
        (; delta, area) = faces[fID]
        snGradP = (P_surff[fID] - P_surf[cID]) / delta
        hfi = hf[fID]
        phif[fID] = rDf[fID] * hfi * hfi * wf[fID] * area * snGradP / rho
    end
end

function remove_film_pressure_source!(U_eqn, P_hydrf, P_surff, rho, h, surfaceNormal, config)
    ∇P_hydr = Grad{Gauss}(P_hydrf)
    ∇P_surf = Grad{Gauss}(P_surff)
    remove_film_pressure_source!(
        U_eqn, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rho, h, surfaceNormal, config
    )
end

function remove_film_pressure_source!(U_eqn, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rho, h, surfaceNormal, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = get_phi(U_eqn).mesh.cells
    (; bx, by, bz) = U_eqn.equation

    grad!(∇P_hydr, P_hydrf, config)
    grad!(∇P_surf, P_surff, config)

    ndrange = length(h)
    kernel! = _remove_film_pressure_source!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, ∇P_hydr, ∇P_surf, rho, h, surfaceNormal, bx, by, bz)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _remove_film_pressure_source!(cells,  ∇P_hydr, ∇P_surf, rho, h, surfaceNormal, bx, by, bz)
    i = @index(Global)

    @inbounds begin
        (; volume) = cells[i]
        n = surfaceNormal[i]
        gradp = ∇P_hydr.result[i] + ∇P_surf.result[i]
        gradp -= (gradp ⋅ n) * n
        source = h[i] * gradp * volume / rho
        bx[i] -= source[1]
        by[i] -= source[2]
        bz[i] -= source[3]
    end
end

function bound_h_nonnegative!(h, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells) = h.mesh
    ndrange = length(cells)

    kernel! = _bound_h_nonnegative!(_setup(backend, workgroup, ndrange)...)
    kernel!(h)
end

@kernel function _bound_h_nonnegative!(h)
    i = @index(Global)

    @inbounds begin
        if h[i] < zero(h[i])
            h[i] = zero(h[i])
        end
    end
end

function update_capillary_dt!(runtime, capillaryDtFaces, mesh, hf, wf, rho, coeffs, config)
    min_capillary_dt = capillary_time_step!(
        capillaryDtFaces, mesh, hf, wf, rho, coeffs.σ, coeffs.h_crit, config
    )
    limit_capillary_dt!(runtime, min_capillary_dt, coeffs)
    return min_capillary_dt
end

function capillary_time_step!(capillaryDtFaces, mesh, hf, wf, rho, σ, h_crit, config)
    if σ <= 0
        return oftype(rho, Inf)
    end

    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(capillaryDtFaces)
    kernel! = _capillary_time_step!(_setup(backend, workgroup, ndrange)...)
    kernel!(capillaryDtFaces, mesh, hf, wf, rho, σ, h_crit, length(mesh.boundary_cellsID))

    return minimum(capillaryDtFaces)
end

@kernel function _capillary_time_step!(capillaryDtFaces, mesh, hf, wf, rho, σ, h_crit, n_bfaces)
    i = @index(Global)

    @inbounds begin
        # Shallow-film capillary CFL from ω² = (σ h / ρ) k⁴, with k = 2π / Δx.
        if i > n_bfaces && wf[i] > zero(wf[i]) && hf[i] > h_crit
            dx = mesh.faces[i].delta
            capillaryDtFaces[i] = sqrt(rho * dx^4 / (4 * pi^2 * σ * max(hf[i], h_crit)))
        else
            capillaryDtFaces[i] = oftype(hf[i], Inf)
        end
    end
end

function limit_capillary_dt!(runtime::Runtime, min_capillary_dt, coeffs)
    if coeffs.σ > 0
        backend = KernelAbstractions.get_backend(runtime.dt)
        kernel! = _limit_capillary_dt!(_setup(backend, 1, 1)...)
        kernel!(runtime.dt, min_capillary_dt, coeffs.capillary_dt)
    end
end

@kernel function _limit_capillary_dt!(dt, min_capillary_dt, capillary_dt)
    i = @index(Global)

    @inbounds begin
        next_dt = dt[i]
        if isfinite(min_capillary_dt) && min_capillary_dt > zero(min_capillary_dt)
            next_dt = min(next_dt, min_capillary_dt)
        end
        if capillary_dt > zero(capillary_dt)
            next_dt = min(next_dt, capillary_dt)
        end
        dt[i] = next_dt
    end
end

function correct_film_velocity!(
    U, Hv, h, P_hydrf, P_surff, rD, rho, surfaceNormal, config;
    include_hydrostatic=true, include_surface=true
)
    if !include_hydrostatic && !include_surface
        copy_projected_vector_field!(U, Hv, surfaceNormal, config)
        return nothing
    end

    ∇P_hydr = Grad{Gauss}(P_hydrf)
    ∇P_surf = Grad{Gauss}(P_surff)
    correct_film_velocity!(
        U, Hv, h, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rD, rho, surfaceNormal, config;
        include_hydrostatic=include_hydrostatic, include_surface=include_surface
    )
end

function correct_film_velocity!(
    U, Hv, h, ∇P_hydr, ∇P_surf, P_hydrf, P_surff, rD, rho, surfaceNormal, config;
    include_hydrostatic=true, include_surface=true
)
    if !include_hydrostatic && !include_surface
        copy_projected_vector_field!(U, Hv, surfaceNormal, config)
        return nothing
    end

    (; hardware) = config
    (; backend, workgroup) = hardware

    if include_hydrostatic
        grad!(∇P_hydr, P_hydrf, config)
    end
    if include_surface
        grad!(∇P_surf, P_surff, config)
    end

    ndrange = length(U)
    kernel! = _correct_film_velocity!(_setup(backend, workgroup, ndrange)...)
    hydrostatic_scale = include_hydrostatic ? one(rho) : zero(rho)
    surface_scale = include_surface ? one(rho) : zero(rho)
    kernel!(U, Hv, h, rD, ∇P_hydr, ∇P_surf, surfaceNormal, rho, hydrostatic_scale, surface_scale)
end

function copy_projected_vector_field!(dest, src, surfaceNormal, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(dest)
    kernel! = _copy_projected_vector_field!(_setup(backend, workgroup, ndrange)...)
    kernel!(dest, src, surfaceNormal)
end

function project_vector_field_to_surface!(field, surfaceNormal, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(field)
    kernel! = _project_vector_field_to_surface!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, surfaceNormal)
end

@kernel function _project_vector_field_to_surface!(field, surfaceNormal)
    i = @index(Global)

    @inbounds begin
        n = surfaceNormal[i]
        value = field[i]
        field[i] = value - (value ⋅ n) * n
    end
end

@kernel function _copy_projected_vector_field!(dest, src, surfaceNormal)
    i = @index(Global)

    @inbounds begin
        n = surfaceNormal[i]
        value = src[i]
        dest[i] = value - (value ⋅ n) * n
    end
end

@kernel function _correct_film_velocity!(
    U, Hv, h, rD, ∇P_hydr, ∇P_surf, surfaceNormal, rho, hydrostatic_scale, surface_scale
)
    i = @index(Global)

    @inbounds begin
        n = surfaceNormal[i]
        pressure_gradient = hydrostatic_scale*∇P_hydr.result[i] + surface_scale*∇P_surf.result[i]
        pressure_gradient -= (pressure_gradient ⋅ n) * n
        value = Hv[i] + pressure_gradient * h[i] * rD[i] / rho
        U[i] = value - (value ⋅ n) * n
    end
end

function correct_film_flux2!(phif, filmSurfaceFlux, Df, h_eqn, w, wf, hBCs, time, config)
    (; mesh) = phif
    (; faces, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    h = h_eqn.model.terms[1].phi
    convection_term = h_eqn.model.terms[2]

    n_bfaces = length(boundary_cellsID)
    n_ifaces = length(faces) - n_bfaces

    apply_film_convection_flux!(phif, convection_term, h, config)

    if n_bfaces > 0
        kernel! = _add_face_flux!(_setup(backend, workgroup, n_bfaces)...)
        kernel!(phif, filmSurfaceFlux)
    end

    if n_ifaces > 0
        kernel! = _correct_film_flux2!(_setup(backend, workgroup, n_ifaces)...)
        kernel!(phif, filmSurfaceFlux, h, Df, faces, n_bfaces)
    end

    for hBC in hBCs
        correct_film_flux_boundary!(phif, h, Df, hBC, time, config)
    end

    apply_donor_wetting_to_flux!(phif, w, wf, config)
    limit_dirichlet_total_flux!(phif, convection_term.flux, hBCs, config)
    apply_film_boundary_flux_policy!(phif, hBCs, config)
end

film_convection_mode(term::Operator{F,P,I,Divergence{Linear}}) where {F,P,I} = 1
film_convection_mode(term::Operator{F,P,I,Divergence{Upwind}}) where {F,P,I} = 2
film_convection_mode(term::Operator{F,P,I,Divergence{LUST}}) where {F,P,I} = 3

function apply_film_convection_flux!(phif, convection_term, h, config)
    (; mesh) = phif
    (; faces, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    n_bfaces = length(boundary_cellsID)
    n_ifaces = length(faces) - n_bfaces

    n_ifaces == 0 && return nothing

    kernel! = _apply_film_convection_flux!(_setup(backend, workgroup, n_ifaces)...)
    kernel!(phif, convection_term.flux, h, faces, n_bfaces, film_convection_mode(convection_term))
end

@kernel function _apply_film_convection_flux!(phif, filmVelocityFlux, h, faces, n_bfaces, mode)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin
        face = faces[fID]
        ownerCells = face.ownerCells
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        flux = filmVelocityFlux[fID]
        h1 = h[cID1]
        h2 = h[cID2]
        h_linear = face.weight*h1 + (one(face.weight) - face.weight)*h2
        h_upwind = ifelse(flux >= zero(flux), h1, h2)
        h_face = ifelse(
            mode == 1,
            h_linear,
            ifelse(mode == 2, h_upwind, 0.75*h_linear + 0.25*h_upwind)
        )
        phif[fID] = flux*h_face
    end
end

@kernel function _add_face_flux!(phif, faceFlux)
    fID = @index(Global)

    @inbounds begin
        phif[fID] += faceFlux[fID]
    end
end

@kernel function _correct_film_flux2!(phif, filmSurfaceFlux, h, Df, faces, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin
        (; ownerCells, delta, area) = faces[fID]

        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        snGrad = (h[cID2] - h[cID1]) / delta

        phif[fID] += filmSurfaceFlux[fID] - Df[fID]*area*snGrad
    end
end

correct_film_flux_boundary!(phif, h, Df, hBC, time, config) = nothing

function correct_film_flux_boundary!(phif, h, Df, hBC::Dirichlet, time, config)
    (; mesh) = phif
    (; faces, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(hBC.IDs_range)
    kernel! = _correct_film_flux_dirichlet_boundary!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, h, Df, faces, boundary_cellsID, hBC.IDs_range, hBC.value)
end

@kernel function _correct_film_flux_dirichlet_boundary!(
    phif, h, Df, faces, boundary_cellsID, IDs_range, value
)
    i = @index(Global)
        fID = IDs_range[i]

    @inbounds begin
        cID = boundary_cellsID[fID]
        (; area, delta) = faces[fID]
        snGrad = (value - h[cID]) / delta
        phif[fID] -= Df[fID] * area * snGrad
    end
end

function max_film_courant_number!(cellsFilmCourant, phif, hf, h_crit, config)
    (; mesh) = phif
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    ndrange = length(cellsFilmCourant)
    kernel! = _max_film_courant_number!(_setup(backend, workgroup, ndrange)...)
    kernel!(cellsFilmCourant, phif, hf, h_crit, runtime, mesh)
    return maximum(cellsFilmCourant)
end

@kernel function _max_film_courant_number!(cellsFilmCourant, phif, hf, h_crit, runtime, mesh)
    cID = @index(Global)

    @uniform begin
        cells = mesh.cells
        cell_faces = mesh.cell_faces
    end

    @inbounds begin
        flux_sum = zero(eltype(cellsFilmCourant))
        for i in cells[cID].faces_range
            fID = cell_faces[i]
            h_scale = max(abs(hf[fID]), h_crit)
            flux_sum += abs(phif[fID]) / h_scale
        end
        cellsFilmCourant[cID] = runtime.dt[1] * flux_sum / cells[cID].volume
    end
end
