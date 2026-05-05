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

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, h, Uf, hf) = model.momentum
    mesh = model.domain
    (; rho) = model.fluid
    

    @info "Pre-allocating fields..."
    rho_mdotf = FaceScalarField(mesh)
    initialise!(rho_mdotf, 0);
    phif = FaceScalarField(mesh)
    nu_h = ScalarField(mesh)
    Sm = ScalarField(mesh)
    divPhi = ScalarField(mesh)
    initialise!(Sm, 0)
    h∇PL = VectorField(mesh)
    Ph = VectorField(mesh)
    τθw = VectorField(mesh)
    Df = FaceScalarField(mesh)
    

    @info "Defining models.."

    # Edit
    @info "U equation still need updating"
    U_eqn = (
        Time{schemes.U.time}(h, U)
        + Divergence{schemes.U.divergence}(phif,U)
        + Si(nu_h, U)
        ==
        - Source(h∇PL)
        + Source(Ph)
        + Source(τθw)
        
    ) → VectorEquation(U, boundaries.U)

    h_eqn = (
        Time{schemes.h.time}(h)
        - Laplacian{schemes.h.laplacian}(Df,h)
        ==
        - Source(divPhi)
        + Source(Sm)
    ) → ScalarEquation(h, boundaries.h)

    @info "Initialising preconditioners"

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset h_eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, h_eqn)

    @info "Pre-allocating solvers"

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset h_eqn.solver = _workspace(solvers.h.solver, _b(h_eqn))

    @info "No turbulence model for now"
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
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; workgroup, backend) = hardware
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware
    
    TF = _get_float(mesh)
    n_cells = length(mesh.cells)
    debug_interval = parse(Int, get(ENV, "XCALIBRE_EFM_DEBUG_INTERVAL", "0"))
    debug_enabled = debug_interval > 0 || get(ENV, "XCALIBRE_EFM_DEBUG", "0") != "0"
    capillaryDtFaces = KernelAbstractions.zeros(backend, TF, length(mesh.faces))
    min_capillary_dt = Inf

    dt_cpu = zeros(TF, 1)
    copyto!(dt_cpu, config.runtime.dt)

    postprocess = convert_time_to_iterations(postprocess, model, dt, iterations)
    phif = get_flux(U_eqn, 2)
    nu_h = get_flux(U_eqn, 3)

    h∇PL = get_source(U_eqn, 1)
    Ph = get_source(U_eqn,2)
    τθw = get_source(U_eqn,3)

    Df = get_flux(h_eqn, 2)
    divPhi = get_source(h_eqn,1)
    Sm = get_source(h_eqn, 2)
    mu = nu.values*rho.values

    outputWriter = initialise_writer(output, model.domain)

    @info "Allocating working memory"

    n = [sind(coeffs.ϕ),0,cosd(coeffs.ϕ)]
    g = 9.8
    G = g.*[0,0,-1]

    # Define aux fields
    hUf = FaceVectorField(mesh)

    mdotf = FaceScalarField(mesh)
    PLf = FaceScalarField(mesh)
    ∇PL = Grad{Gauss}(PLf)
    P_gasf = FaceScalarField(mesh)
    P_hydr = ScalarField(mesh)
    P_hydrf = FaceScalarField(mesh)
    P_surf = ScalarField(mesh)
    P_surff = FaceScalarField(mesh)
    Pf = FaceScalarField(mesh)

    tempU = VectorField(mesh)
    Δh = ScalarField(mesh)
    Δhf = FaceScalarField(mesh)

    w = ScalarField(mesh)
    wf = FaceScalarField(mesh)
    ∇w = Grad{schemes.h.gradient}(w)

    Hv = VectorField(mesh)
    HbyA = VectorField(mesh)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)
    
    # Current film meshes use x as the downslope surface coordinate.
    plate_tangent_vector = Vector{}([1,0,0])
    gravity_tangent = (g*sind(coeffs.ϕ)) .* plate_tangent_vector
   
    # u_inlet = boundaries.U[1].value
    # h_inlet = boundaries.h[1].value
    # hU_inlet = [u_inlet[1] .* h_inlet, u_inlet[2] .* h_inlet, u_inlet[3] .* h_inlet]

    internal_BCs = assign(
        region=mesh,
        (
            Δh = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            Extrapolated(:inlet_sides),
            Extrapolated(:top_of_plate),
            Extrapolated(:side_1),
            Extrapolated(:side_2)
            ],
            w = [
            Dirichlet(:inlet, 1),
            Zerogradient(:outlet),
            Zerogradient(:inlet_sides),
            Zerogradient(:top_of_plate),
            Zerogradient(:side_1),
            Zerogradient(:side_2)
    ]
    #         w = [
    #         Dirichlet(:inlet, 1),
    #         Extrapolated(:outlet),
    #         Extrapolated(:inlet_sides),
    #         Extrapolated(:top_of_plate),
    #         Extrapolated(:side_1),
    #         Extrapolated(:side_2)
    # ]
        )
    )

    # Pre-allocate auxiliary variables
    h_old = KernelAbstractions.zeros(backend, TF, n_cells)
    prev = KernelAbstractions.zeros(backend, TF, n_cells)

    # Pre-allocate vectors to hold residuals
    R_ux = zeros(TF, iterations)
    R_uy = zeros(TF, iterations)
    R_uz = zeros(TF, iterations)
    R_h = zeros(TF, iterations)
    courant = zeros(TF, iterations)
    cellsCourant = KernelAbstractions.zeros(backend, TF, n_cells)
    cellsFilmCourant = KernelAbstractions.zeros(backend, TF, n_cells)

    # Initial calculations
    time = zero(TF) # assuming time = 0

    limit_h!(h, coeffs.h_floor, config)

    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    update_wetting_fields!(w, wf, h, internal_BCs.w, coeffs.h_crit, time, config)
    set_fixed_h_boundary_cells_wet!(w, wf, boundaries.h, internal_BCs.w, time, config)
    
    # Getting the laplacian of h for first U calculation
    laplacian!(Δh, hf, h, boundaries.h, time, config)
    interpolate!(Δhf, Δh, config)
    correct_boundaries!(Δhf, Δh, internal_BCs.Δh, time, config)

    apply_film_flux!(phif, mdotf, h, hf, wf, coeffs.h_crit, config)
    min_capillary_dt = update_capillary_dt!(
        config.runtime, capillaryDtFaces, mesh, hf, wf, rho.values[1], coeffs, config
    )
    copyto!(dt_cpu, config.runtime.dt)

    #@info "need to readd Pg term - Coupling term for other phase"
    Pg = 0# Test Pg term set to zero, as the gradient is found this value doesn't matter

    @. nu_h.values = 3*nu.values/h.values
    update_film_pressure_fields!(
        Δh, hf, Δhf, P_hydr, P_hydrf, P_surf, P_surff, h,
        boundaries.h, internal_BCs.Δh, rho.values[1],
        coeffs.σ, G, n, time, config
    )

    for i ∈ eachindex(Δhf.values)
        P_gasf.values[i] = Pg
        PLf[i] = P_gasf[i] - P_hydrf[i] - P_surff[i]
    end

    grad!(∇PL, PLf, config)

    grad!(∇w, wf, w, internal_BCs.w, time, config)

    update_film_force_sources!(
        Ph, h∇PL, τθw, h, ∇PL, ∇w, gravity_tangent,
        plate_tangent_vector, coeffs, rho.values[1], config
    )

    @info "Starting loops"
    
    progress = Progress(iterations; dt=1.0, showspeed=true)
            
    xdir, ydir, zdir = XDir(), YDir(), ZDir()
    #rh = 0
    rx = ry = rz = 0
    last_flux_correction_max = zero(TF)
    @time for iteration ∈ 1:iterations
        min_capillary_dt = update_capillary_dt!(
            config.runtime, capillaryDtFaces, mesh, hf, wf, rho.values[1], coeffs, config
        )
        copyto!(dt_cpu, config.runtime.dt)
        time += dt_cpu[1]
        step_dt = dt_cpu[1]

        @. h_old = h.values # store previous h before inner loop

        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config, time=time)
        
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_film_pressure_source!(U_eqn, P_hydrf, P_surff, rho.values[1], h, config)

        H!(Hv, U, U_eqn, config)
        correct_film_velocity!(
            tempU, Hv, h, P_hydrf, P_surff, rD, rho.values[1],
            config; include_hydrostatic=false, include_surface=false
        )
        rh = 0
        for i ∈ 1:inner_loops
            H!(Hv, tempU, U_eqn, config)

            update_film_pressure_fields!(
                Δh, hf, Δhf, P_hydr, P_hydrf, P_surf, P_surff, h,
                boundaries.h, internal_BCs.Δh, rho.values[1],
                coeffs.σ, G, n, time, config
            )

            correct_film_velocity!(
                tempU, Hv, h, P_hydrf, P_surff, rD, rho.values[1],
                config; include_hydrostatic=false, include_surface=false
            )
            interpolate!(Uf, tempU, config)
            correct_boundaries!(Uf, tempU, boundaries.U, time, config)
            flux!(mdotf, Uf, config)
            apply_film_flux!(phif, mdotf, h, hf, wf, coeffs.h_crit, config)

            getDf!(Df, rDf, hf, wf, G, n, config)
            correct_film_surface_flux!(
                phif, rDf, hf, wf, P_surf, P_surff, rho.values[1], config
            )

            div!(divPhi, phif, config)
            
            @. prev = h.values
            rh = solve_film_h_equation!(h_eqn, h, h_old, boundaries.h, solvers.h, config, time=time)

            if i == inner_loops
                explicit_relaxation!(h, prev, 1.0, config)
            else
                explicit_relaxation!(h, prev, solvers.h.relax, config)
            end 

            limit_h!(h, coeffs.h_floor, config)
            update_wetting_fields!(w, wf, h, internal_BCs.w, coeffs.h_crit, time, config)
            update_contact_line_source!(
                τθw, ∇w, wf, w, internal_BCs.w, coeffs,
                rho.values[1], plate_tangent_vector, time, config
            )
            
            update_film_pressure_fields!(
                Δh, hf, Δhf, P_hydr, P_hydrf, P_surf, P_surff, h,
                boundaries.h, internal_BCs.Δh, rho.values[1],
                coeffs.σ, G, n, time, config
            )
            getDf!(Df, rDf, hf, wf, G, n, config)

            correct_film_velocity!(
                U, Hv, h, P_hydrf, P_surff, rD, rho.values[1],
                config; include_hydrostatic=true
            )
            correct_film_velocity!(
                tempU, Hv, h, P_hydrf, P_surff, rD, rho.values[1],
                config; include_hydrostatic=false, include_surface=false
            )
            interpolate!(Uf, tempU, config)
            correct_boundaries!(Uf, tempU, boundaries.U, time, config)
            flux!(mdotf, Uf, config)
            apply_film_flux!(phif, mdotf, h, hf, wf, coeffs.h_crit, config)
            correct_film_surface_flux!(
                phif, rDf, hf, wf, P_surf, P_surff, rho.values[1], config
            )
            if debug_enabled && (debug_interval <= 0 || iteration == 1 || iteration % debug_interval == 0)
                last_flux_correction_max = max_film_flux_correction(Df, h, config)
            end
            correct_film_flux2!(phif, Df, h_eqn, boundaries.h, time, config)
        end

        @. nu_h.values = 3*nu.values/h.values
        for i ∈ eachindex(Δhf.values)
            P_gasf.values[i] = Pg

            PLf[i] = P_gasf[i] - P_hydrf[i] - P_surff[i]
        end

        grad!(∇PL, PLf, config)

        update_wetting_fields!(w, wf, h, internal_BCs.w, coeffs.h_crit, time, config)

        # correct U for non-wetted
        for i ∈ eachindex(U.x)
            U[i] = U[i] .* w[i]
        end

        correct_film_velocity!(
            tempU, Hv, h, P_hydrf, P_surff, rD, rho.values[1],
            config; include_hydrostatic=false, include_surface=false
        )
        interpolate!(Uf, tempU, config)
        correct_boundaries!(Uf, tempU, boundaries.U, time, config)
        flux!(mdotf, Uf, config)
        apply_film_flux!(phif, mdotf, h, hf, wf, coeffs.h_crit, config)
        getDf!(Df, rDf, hf, wf, G, n, config)
        correct_film_surface_flux!(
            phif, rDf, hf, wf, P_surf, P_surff, rho.values[1], config
        )
        correct_film_flux2!(phif, Df, h_eqn, boundaries.h, time, config)

        grad!(∇w, wf, w, internal_BCs.w, time, config)
        update_film_force_sources!(
            Ph, h∇PL, τθw, h, ∇PL, ∇w, gravity_tangent,
            plate_tangent_vector, coeffs, rho.values[1], config
        )

        #for i ∈ 1:ncorrectors
        #    discretise!(h_eqn, h, config)
        #    apply_boundary_conditions!(h_eqn, boundaries.h, nothing, time, config)
  
        #    rh = solve_system!(h_eqn, solvers.h, h, nothing, config)
        #    explicit_relaxation!(h, prev, solvers.h.relax, config)
        #end
        
        
        R_ux[iteration] = rx
        R_uy[iteration] = ry
        R_uz[iteration] = rz
        R_h[iteration] = rh



        maxCourant = max_courant_number!(cellsCourant, model, config)
        maxFilmCourant = max_film_courant_number!(cellsFilmCourant, phif, hf, coeffs.h_crit, config)
        limitingCourant = max(maxCourant, maxFilmCourant)
        maxCourantDt = courant_dt_limit(maxCourant, step_dt, config.runtime)
        maxFilmCourantDt = courant_dt_limit(maxFilmCourant, step_dt, config.runtime)
        courant[iteration] = limitingCourant
        update_dt!(config.runtime, limitingCourant)
        min_capillary_dt = update_capillary_dt!(
            config.runtime, capillaryDtFaces, mesh, hf, wf, rho.values[1], coeffs, config
        )
        maxStableDt = min(maxCourantDt, maxFilmCourantDt, min_capillary_dt)
        copyto!(dt_cpu, config.runtime.dt)
        next_dt = dt_cpu[1]

        ProgressMeter.next!(
            progress, showvalues = [
                (:dt, step_dt),
                (Symbol("next dt"), next_dt),
                (Symbol("max dt (Courant)"), maxCourantDt),
                (Symbol("max dt (Film Courant)"), maxFilmCourantDt),
                (Symbol("max dt (capillary)"), min_capillary_dt),
                (Symbol("max dt (stable)"), maxStableDt),
                (:time, time),
                (:Courant, maxCourant),
                (Symbol("Film Courant"), maxFilmCourant),
                (Symbol("Limiting Courant"), limitingCourant),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:h, R_h[iteration]),
                #turbulenceModel.state.residuals...
            ]
        )

        #runtime_postprocessing!(postprocess, iteration, iterations)

        ∇P_hydr = Grad{Gauss}(P_hydrf)
        grad!(∇P_hydr, P_hydrf, config)
        ∇P_surf = Grad{Gauss}(P_surff)
        grad!(∇P_surf, P_surff, config)

        if debug_enabled && (debug_interval <= 0 || iteration == 1 || iteration % debug_interval == 0)
            print_film_diagnostics!(
                cellsFilmCourant, iteration, time, step_dt, h, U, phif, hf, coeffs.h_crit, w, wf,
                P_surff, ∇P_surf.result, last_flux_correction_max, config
            )
        end

        if iteration % write_interval + signbit(write_interval) == 0

            save_output_film(model, outputWriter, iteration, time, config, w)
            # More verbose output for extra details for debugging
            #save_output_film(model, outputWriter, iteration, time, config, w, Δh, h∇PL, nu_h, Ph, τθw, divPhi, tempU, Hv, ∇P_hydr.result, P_hydr, ∇h.result, ∇P_surf.result)
            save_postprocessing(postprocess, iteration, time, mesh, outputWriter, config.boundaries)
        end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, h=R_h, courant=courant)
end

function update_film_pressure_fields!(
    Δh, hf, Δhf, P_hydr, P_hydrf, P_surf, P_surff, h,
    hBCs, ΔhBCs, rho, σ, G, n, time, config
)
    laplacian!(Δh, hf, h, hBCs, time, config, disp_warn=false)
    interpolate!(Δhf, Δh, config)
    correct_boundaries!(Δhf, Δh, ΔhBCs, time, config)

    (; hardware) = config
    (; backend, workgroup) = hardware

    g_n = dot(n, G)
    cell_kernel! = _update_film_pressure_cells!(_setup(backend, workgroup, length(h))...)
    cell_kernel!(P_hydr, P_surf, h, Δh, rho, σ, g_n)

    face_kernel! = _update_film_pressure_faces!(_setup(backend, workgroup, length(hf))...)
    face_kernel!(P_hydrf, P_surff, hf, Δhf, rho, σ, g_n)
end

@kernel function _update_film_pressure_cells!(P_hydr, P_surf, h, Δh, rho, σ, g_n)
    i = @index(Global)

    @inbounds begin
        P_hydr[i] = rho * h[i] * g_n
        P_surf[i] = σ * Δh[i]
    end
end

@kernel function _update_film_pressure_faces!(P_hydrf, P_surff, hf, Δhf, rho, σ, g_n)
    i = @index(Global)

    @inbounds begin
        P_hydrf[i] = rho * hf[i] * g_n
        P_surff[i] = σ * Δhf[i]
    end
end

function update_film_force_sources!(
    Ph, h∇PL, τθw, h, ∇PL, ∇w, gravity_tangent,
    plate_tangent_vector, coeffs, rho, config
)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(h)
    kernel! = _update_film_force_sources!(_setup(backend, workgroup, ndrange)...)
    contact_line_scale = coeffs.β * coeffs.σ / rho * (1 - cosd(coeffs.θm))
    kernel!(
        Ph, h∇PL, τθw, h, ∇PL, ∇w, gravity_tangent,
        plate_tangent_vector, contact_line_scale, rho
    )
end

@kernel function _update_film_force_sources!(
    Ph, h∇PL, τθw, h, ∇PL, ∇w, gravity_tangent,
    plate_tangent_vector, contact_line_scale, rho
)
    i = @index(Global)

    @uniform begin
        Phx, Phy, Phz = Ph.x, Ph.y, Ph.z
        h∇PLx, h∇PLy, h∇PLz = h∇PL.x, h∇PL.y, h∇PL.z
        τθwx, τθwy, τθwz = τθw.x, τθw.y, τθw.z
        dPLdx, dPLdy, dPLdz = ∇PL.result.x, ∇PL.result.y, ∇PL.result.z
        dwdx, dwdy, dwdz = ∇w.result.x, ∇w.result.y, ∇w.result.z
    end

    @inbounds begin
        hi = h[i]

        Phx[i] = hi * gravity_tangent[1]
        Phy[i] = hi * gravity_tangent[2]
        Phz[i] = hi * gravity_tangent[3]

        h∇PLx[i] = hi * dPLdx[i] / rho
        h∇PLy[i] = hi * dPLdy[i] / rho
        h∇PLz[i] = hi * dPLdz[i] / rho

        grad_dot_t = dwdx[i] * plate_tangent_vector[1] +
                     dwdy[i] * plate_tangent_vector[2] +
                     dwdz[i] * plate_tangent_vector[3]
        τθwx[i] = contact_line_scale * (dwdx[i] - grad_dot_t * plate_tangent_vector[1])
        τθwy[i] = contact_line_scale * (dwdy[i] - grad_dot_t * plate_tangent_vector[2])
        τθwz[i] = contact_line_scale * (dwdz[i] - grad_dot_t * plate_tangent_vector[3])
    end
end

function update_wetting_fields!(w, wf, h, wBCs, h_crit, time, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(w)
    wetting_mode = efm_wetting_mode()
    if wetting_mode === :allwet
        kernel! = _set_wetting_field!(_setup(backend, workgroup, ndrange)...)
        kernel!(w, one(h_crit))
    elseif wetting_mode === :smooth
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

const _efm_unknown_wetting_mode = Ref("")

function efm_wetting_mode()
    mode = lowercase(strip(get(ENV, "XCALIBRE_EFM_WETTING", "hard")))

    if mode == "hard"
        return :hard
    elseif mode == "smooth" || mode == "smoothed"
        return :smooth
    elseif mode == "allwet"
        return :allwet
    end

    if _efm_unknown_wetting_mode[] != mode
        @warn "Unknown XCALIBRE_EFM_WETTING mode; falling back to hard" mode
        _efm_unknown_wetting_mode[] = mode
    end
    return :hard
end

function set_fixed_h_boundary_cells_wet!(w, wf, hBCs, wBCs, time, config)
    for hBC in hBCs
        set_fixed_h_boundary_cells_wet!(w, hBC, config)
    end

    interpolate!(wf, w, config)
    correct_boundaries!(wf, w, wBCs, time, config)
    clamp_face_wetting!(wf, config)
end

set_fixed_h_boundary_cells_wet!(w, hBC, config) = nothing

function set_fixed_h_boundary_cells_wet!(w, hBC::AbstractDirichlet, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; boundary_cellsID) = w.mesh

    ndrange = length(hBC.IDs_range)
    kernel! = _set_fixed_h_boundary_cells_wet!(_setup(backend, workgroup, ndrange)...)
    kernel!(w, boundary_cellsID, hBC.IDs_range)
end

@kernel function _set_fixed_h_boundary_cells_wet!(w, boundary_cellsID, IDs_range)
    i = @index(Global)

    @inbounds begin
        cID = boundary_cellsID[IDs_range[i]]
        w[cID] = one(w[cID])
    end
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

function apply_film_flux!(phif, mdotf, h, hf, wf, h_crit, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; faces, boundary_cellsID) = phif.mesh

    ndrange = length(phif)
    kernel! = _apply_film_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, mdotf, h, hf, wf, h_crit, faces, length(boundary_cellsID))
end

@kernel function _apply_film_flux!(phif, mdotf, h, hf, wf, h_crit, faces, n_bfaces)
    fID = @index(Global)

    @inbounds begin
        h_interp = hf[fID]
        h_face = h_interp
        if fID > n_bfaces
            face = faces[fID]
            ownerCells = face.ownerCells
            h1 = h[ownerCells[1]]
            h2 = h[ownerCells[2]]
            wet1 = h1 > h_crit
            wet2 = h2 > h_crit
            h_upwind = ifelse(mdotf[fID] >= zero(mdotf[fID]), h1, h2)
            lateral_face = abs(face.normal[2]) >= max(abs(face.normal[1]), abs(face.normal[3]))
            h_face = ifelse(wet1 == wet2 || !lateral_face, h_upwind, h_interp)
        end
        phif[fID] = mdotf[fID] * h_face * wf[fID]
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


# Reworked save_output for film model
function save_output_film(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu,E,D,BI}
    args = (
            ("U", model.momentum.U), 
            ("h", model.momentum.h),
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
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

function save_output_film(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config, w, Δh, h∇PL, nu_h, Ph, τθw, divPhi, tempU, Hv, ∇P_hydr, P_hydr, ∇h, ∇P_surf
    ) where {T,F,SO,M,Tu,E,D,BI}

    mesh = w.mesh
    Cids = ScalarField(mesh)
    for i ∈ eachindex(mesh.cells)
        Cids[i] = i
    end

    args = (
            ("U", model.momentum.U), 
            ("h", model.momentum.h),
            ("w", w),
            ("Δh", Δh),
            ("h∇PL", h∇PL),
            ("nu_h", nu_h),
            ("Ph", Ph),
            ("τθw", τθw),
            ("divPhi", divPhi),
            ("cID", Cids),
            ("U_temp", tempU),
            ("Hv", Hv),
            ("P_hydr", P_hydr),
            ("∇P_hydr", ∇P_hydr),
            ("∇h", ∇h),
            ("∇P_surf", ∇P_surf)
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

function getDf!(Df, rDf, hf, wf, g, n, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    ndrange = length(hf)
    kernel! = _getDf!(_setup(backend, workgroup, ndrange)...)
    kernel!(Df, rDf, hf, wf, g, n)
end

@kernel function _getDf!(Df, rDf, hf, wf, g, n)
    i = @index(Global)

    @inbounds begin
        g_n = dot(g, n)
        Df[i] = -rDf[i] * hf[i]^2 * wf[i] * g_n
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

@kernel function _correct_film_surface_flux!(phif, rDf, hf, wf, P_surf, faces, rho, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin
        (; ownerCells, delta, area) = faces[fID]
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        snGradP = (P_surf[cID2] - P_surf[cID1]) / delta
        phif[fID] += rDf[fID] * hf[fID]^2 * wf[fID] * area * snGradP / rho
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
        phif[fID] += rDf[fID] * hf[fID]^2 * wf[fID] * area * snGradP / rho
    end
end

function update_contact_line_source!(τθw, ∇w, wf, w, wBCs, coeffs, rho, plate_tangent_vector, time, config)
    grad!(∇w, wf, w, wBCs, time, config)

    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(w)
    kernel! = _update_contact_line_source!(_setup(backend, workgroup, ndrange)...)
    contact_line_scale = coeffs.β * coeffs.σ / rho * (1 - cosd(coeffs.θm))
    kernel!(τθw, ∇w, plate_tangent_vector, contact_line_scale)
end

@kernel function _update_contact_line_source!(τθw, ∇w, plate_tangent_vector, contact_line_scale)
    i = @index(Global)

    @uniform begin
        τθwx, τθwy, τθwz = τθw.x, τθw.y, τθw.z
        dwdx, dwdy, dwdz = ∇w.result.x, ∇w.result.y, ∇w.result.z
    end

    @inbounds begin
        grad_dot_t = dwdx[i] * plate_tangent_vector[1] +
                     dwdy[i] * plate_tangent_vector[2] +
                     dwdz[i] * plate_tangent_vector[3]
        τθwx[i] = contact_line_scale * (dwdx[i] - grad_dot_t * plate_tangent_vector[1])
        τθwy[i] = contact_line_scale * (dwdy[i] - grad_dot_t * plate_tangent_vector[2])
        τθwz[i] = contact_line_scale * (dwdz[i] - grad_dot_t * plate_tangent_vector[3])
    end
end

function remove_film_pressure_source!(U_eqn, P_hyrdf, P_surff, rho, h, config)
    
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = get_phi(U_eqn).mesh.cells
    (; bx, by, bz) = U_eqn.equation

    ∇P_hydr = Grad{Gauss}(P_hyrdf)
    ∇P_surf = Grad{Gauss}(P_surff)

    grad!(∇P_hydr, P_hyrdf, config)
    grad!(∇P_surf, P_surff, config)

    ndrange = length(h)
    kernel! = _remove_film_pressure_source!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, ∇P_hydr, ∇P_surf, rho, h, bx, by, bz)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _remove_film_pressure_source!(cells,  ∇P_hydr, ∇P_surf, rho, h, bx, by, bz)
    i = @index(Global)

    @uniform begin
        ∇P_hydr_x, ∇P_hydr_y, ∇P_hydr_z = ∇P_hydr.result.x, ∇P_hydr.result.y, ∇P_hydr.result.z
        ∇P_surf_x, ∇P_surf_y, ∇P_surf_z = ∇P_surf.result.x, ∇P_surf.result.y, ∇P_surf.result.z
        _h = h
    end

    @inbounds begin
        hi = _h[i]
        (; volume) = cells[i]
        bx[i] -= hi*(∇P_hydr_x[i] + ∇P_surf_x[i])*volume/rho
        by[i] -= hi*(∇P_hydr_y[i] + ∇P_surf_y[i])*volume/rho
        bz[i] -= hi*(∇P_hydr_z[i] + ∇P_surf_z[i])*volume/rho
    end
end

function limit_h!(h, h_floor, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells) = h.mesh
    ndrange = length(cells)

    kernel! = _limit_h!(_setup(backend, workgroup, ndrange)...)
    kernel!(h, h_floor)
end

@kernel function _limit_h!(h, h_floor)
    i = @index(Global)

    @inbounds begin
        if (h[i] <= h_floor) h[i] = h_floor end
    end
end

function update_capillary_dt!(runtime, capillaryDtFaces, mesh, hf, wf, rho, coeffs, config)
    min_capillary_dt = capillary_time_step!(
        capillaryDtFaces, mesh, hf, wf, rho, coeffs.σ, coeffs.h_crit, config
    )
    limit_capillary_dt!(runtime, min_capillary_dt, coeffs)
    return min_capillary_dt
end

function courant_dt_limit(courant, dt, runtime)
    target = courant_target(runtime)
    return courant > 0 ? dt * target / courant : oftype(dt, Inf)
end

courant_target(runtime::Runtime{<:Any,<:Any,<:Any,Nothing}) = one(eltype(runtime.dt))
courant_target(runtime::Runtime{<:Any,<:Any,<:Any,<:AdaptiveTimeStepping}) = runtime.adaptive.maxCo

function capillary_time_step!(capillaryDtFaces, mesh::Mesh2, hf, wf, rho, σ, h_crit, config)
    if σ <= 0
        return oftype(rho, Inf)
    end

    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(capillaryDtFaces)
    kernel! = _capillary_time_step_2d!(_setup(backend, workgroup, ndrange)...)
    kernel!(capillaryDtFaces, mesh, hf, wf, rho, σ, h_crit, length(mesh.boundary_cellsID))

    return minimum(capillaryDtFaces)
end

capillary_time_step!(capillaryDtFaces, mesh, hf, wf, rho, σ, h_crit, config) = oftype(rho, Inf)

@kernel function _capillary_time_step_2d!(capillaryDtFaces, mesh, hf, wf, rho, σ, h_crit, n_bfaces)
    i = @index(Global)

    @inbounds begin
        if i > n_bfaces && wf[i] >= one(wf[i]) && hf[i] > h_crit
            dx = mesh.faces[i].delta
            capillaryDtFaces[i] = sqrt(rho * dx^3 / (2 * pi * σ))
        else
            capillaryDtFaces[i] = oftype(hf[i], Inf)
        end
    end
end

function limit_capillary_dt!(runtime::Runtime, min_capillary_dt, coeffs)
    if coeffs.σ > 0
        if isfinite(min_capillary_dt) && min_capillary_dt > 0
            runtime.dt[1] = min(runtime.dt[1], min_capillary_dt)
        end
        if coeffs.capillary_dt > 0
            runtime.dt[1] = min(runtime.dt[1], coeffs.capillary_dt)
        end
    end
end

function correct_film_velocity!(
    U, Hv, h, P_hydrf, P_surff, rD, rho, config;
    include_hydrostatic=true, include_surface=true
)
    (; mesh) = U
    (; hardware) = config
    (; backend, workgroup) = hardware

    ∇P_hydr = Grad{Gauss}(P_hydrf)
    grad!(∇P_hydr, P_hydrf, config)
    ∇P_surf = Grad{Gauss}(P_surff)
    grad!(∇P_surf, P_surff, config)

    ndrange = length(U)
    kernel! = _correct_film_velocity!(_setup(backend, workgroup, ndrange)...)
    hydrostatic_scale = include_hydrostatic ? one(rho) : zero(rho)
    surface_scale = include_surface ? one(rho) : zero(rho)
    kernel!(U, Hv, h, rD, ∇P_hydr, ∇P_surf, rho, hydrostatic_scale, surface_scale)
end

@kernel function _correct_film_velocity!(
    U, Hv, h, rD, ∇P_hydr, ∇P_surf, rho, hydrostatic_scale, surface_scale
)
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hvx, Hvy, Hvz = Hv.x, Hv.y, Hv.z
        _h = h
        dPhdx, dPhdy, dPhdz = ∇P_hydr.result.x, ∇P_hydr.result.y, ∇P_hydr.result.z
        dPsdx, dPsdy, dPsdz = ∇P_surf.result.x, ∇P_surf.result.y, ∇P_surf.result.z
        _rD = rD
    end

    @inbounds begin
        rDi = _rD[i]
        hi = _h[i]
        Ux[i] = Hvx[i] + (hydrostatic_scale*dPhdx[i] + surface_scale*dPsdx[i]) * hi * rDi/rho
        Uy[i] = Hvy[i] + (hydrostatic_scale*dPhdy[i] + surface_scale*dPsdy[i]) * hi * rDi/rho
        Uz[i] = Hvz[i] + (hydrostatic_scale*dPhdz[i] + surface_scale*dPsdz[i]) * hi * rDi/rho
    end
end

function correct_film_flux2!(phif, Df, h_eqn, hBCs, time, config)
    (; mesh) = phif
    (; faces, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    h = h_eqn.model.terms[1].phi

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces
    kernel! = _correct_film_flux2!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, h, Df, faces, n_bfaces)

    for hBC in hBCs
        correct_film_flux_boundary!(phif, h, Df, hBC, time, config)
    end
end

@kernel function _correct_film_flux2!(phif, h, Df, faces, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    (; ownerCells, delta, area) = faces[fID]

    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    snGrad = (h[cID2] - h[cID1])/delta

    phif[fID] -= Df[fID]*area*snGrad
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

function max_film_flux_correction(Df, h, config)
    (; mesh) = Df
    (; faces, boundary_cellsID) = mesh
    Df_values = _cpu_values(Df.values)
    h_values = _cpu_values(h.values)
    n_bfaces = length(boundary_cellsID)
    max_correction = zero(eltype(Df_values))
    for fID in (n_bfaces + 1):length(faces)
        (; ownerCells, delta, area) = faces[fID]
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        snGrad = (h_values[cID2] - h_values[cID1]) / delta
        max_correction = max(max_correction, abs(Df_values[fID] * area * snGrad))
    end
    return max_correction
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

function _cpu_values(values)
    cpu = zeros(eltype(values), length(values))
    copyto!(cpu, values)
    return cpu
end

function _max_vector_magnitude_cpu(U)
    ux = _cpu_values(U.x.values)
    uy = _cpu_values(U.y.values)
    uz = _cpu_values(U.z.values)
    max_u = zero(eltype(ux))
    sum_u = zero(eltype(ux))
    for i in eachindex(ux)
        mag = sqrt(ux[i]^2 + uy[i]^2 + uz[i]^2)
        max_u = max(max_u, mag)
        sum_u += mag
    end
    return max_u, sum_u / length(ux)
end

function _max_film_courant_cpu(phif, hf, h_crit, dt)
    (; mesh) = phif
    (; cells, cell_faces) = mesh
    phif_values = _cpu_values(phif.values)
    hf_values = _cpu_values(hf.values)
    max_film_co = zero(eltype(phif_values))
    for cID in eachindex(cells)
        flux_sum = zero(eltype(phif_values))
        for i in cells[cID].faces_range
            fID = cell_faces[i]
            h_scale = max(abs(hf_values[fID]), h_crit)
            flux_sum += abs(phif_values[fID]) / h_scale
        end
        max_film_co = max(max_film_co, dt * flux_sum / cells[cID].volume)
    end
    return max_film_co
end

function print_film_diagnostics!(
    cellsFilmCourant, iteration, time, dt, h, U, phif, hf, h_crit, w, wf, P_surff, ∇P_surf,
    flux_correction_abs_max, config
)
    h_values = _cpu_values(h.values)
    w_values = _cpu_values(w.values)
    wf_values = _cpu_values(wf.values)
    ps_values = _cpu_values(P_surff.values)
    max_u, mean_u = _max_vector_magnitude_cpu(U)
    max_film_co = _max_film_courant_cpu(phif, hf, h_crit, dt)
    max_grad_ps, mean_grad_ps = _max_vector_magnitude_cpu(∇P_surf)

    wet_cells = count(>(0), w_values)
    partial_faces = count(x -> x > 0 && x < 1, wf_values)
    @info "EFM diagnostics" iteration time dt h_min=minimum(h_values) h_max=maximum(h_values) h_mean=sum(h_values)/length(h_values) U_max=max_u U_mean=mean_u film_Co_max=max_film_co wet_cells wet_fraction=wet_cells/length(w_values) wf_min=minimum(wf_values) wf_max=maximum(wf_values) wf_partial_fraction=partial_faces/length(wf_values) Psurf_abs_max=maximum(abs, ps_values) gradPsurf_abs_max=max_grad_ps gradPsurf_abs_mean=mean_grad_ps flux_correction_abs_max
end
