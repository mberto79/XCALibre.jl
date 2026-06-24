export multiphase!

function multiphase!(
    model, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2)

    residuals = setup_multiphase_solvers(
        MULTIPHASE, model, config;
        output=output,
        pref=pref,
        ncorrectors=ncorrectors,
        inner_loops=inner_loops
        )

    return residuals
end

multiphase_extras(::VOF, mesh) = ()

function multiphase_extras(::Mixture, mesh)
    div_slip_momentum = VectorField(mesh)
    return (div_slip_momentum,)
end

function setup_multiphase_solvers(
    solver_variant, model, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, p) = model.momentum
    (; alpha, alphaf, rho, rhof, nu, nuf, p_rgh, p_rghf) = model.fluid

    mp_model = model.fluid.model

    phases = model.fluid.phases
    volume_fraction = model.fluid.volume_fraction
    main = volume_fraction
    secondary = 3 - volume_fraction

    mesh = model.domain

    @info "Pre-allocating fields..."

    TF = _get_float(mesh)
    time = zero(TF)

    ∇p = Grad{schemes.p_rgh.gradient}(p)

    ∇p_rgh = Grad{schemes.p_rgh.gradient}(p_rgh)
    grad!(∇p_rgh, p_rghf, p_rgh, boundaries.p_rgh, time, config)
    limit_gradient!(schemes.p_rgh.limiter, ∇p_rgh, p_rgh, config)

    mdotf = FaceScalarField(mesh)
    rhoPhi = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    mueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)

    phi_g = VectorField(mesh)
    phi_gf = FaceScalarField(mesh)

    extra_models = multiphase_extras(mp_model, mesh)

    mules = (
        alpha_prev    = ScalarField(mesh),

        div_alpha     = ScalarField(mesh),
        div_mdotf     = ScalarField(mesh),

        alpha_fluxf   = FaceScalarField(mesh),
        alphaf_upwind = FaceScalarField(mesh),
        alphaf_HO     = FaceScalarField(mesh),

        phiLf         = FaceScalarField(mesh),
        phiHf         = FaceScalarField(mesh),
        phiAf         = FaceScalarField(mesh),

        Pplus         = ScalarField(mesh),
        Pminus        = ScalarField(mesh),

        Qplus         = ScalarField(mesh),
        Qminus        = ScalarField(mesh),

        Rplus         = ScalarField(mesh),
        Rminus        = ScalarField(mesh),

        alphaMaxLocal = ScalarField(mesh),
        alphaMinLocal = ScalarField(mesh),
    )

    @info "Computing fluid properties..."

    blend_properties!(rho, alpha, phases[main].rho[1], phases[secondary].rho[1])
    blend_properties!(rhof, alphaf, phases[main].rho[1], phases[secondary].rho[1])
    blend_properties!(nuf, alphaf, phases[main].mu[1] / phases[main].rho[1], phases[secondary].mu[1] / phases[secondary].rho[1])
    @. mueff.values = rhof.values * nueff.values

    gh = model.fluid.physics_properties.gravity.gh
    ghf = model.fluid.physics_properties.gravity.ghf
    g = model.fluid.physics_properties.gravity.g

    compute_gh!(gh, g, config)
    compute_ghf!(ghf, g, config)

    @info "Defining models..."

    if typeof(mp_model) <: VOF

        U_eqn = (
            Time{schemes.U.time}(rho, U)
            + Divergence{schemes.U.divergence}(rhoPhi, U)
            - Laplacian{schemes.U.laplacian}(mueff, U)
            ==
            - Source(∇p_rgh.result)
        ) → VectorEquation(U, boundaries.U)

    elseif typeof(mp_model) <: Mixture

        div_slip_momentum = extra_models[1]

        U_eqn = (
            Time{schemes.U.time}(rho, U)
            + Divergence{schemes.U.divergence}(rhoPhi, U)
            - Laplacian{schemes.U.laplacian}(mueff, U)
            ==
            - Source(∇p_rgh.result)
            - Source(div_slip_momentum)
        ) → VectorEquation(U, boundaries.U)

    end

    p_eqn = (
        - Laplacian{schemes.p.laplacian}(rDf, p_rgh)
        ==
        - Source(divHv)
    ) → ScalarEquation(p_rgh, boundaries.p_rgh)

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p_rgh.preconditioner, p_eqn)

    @info "Pre-allocating solvers..."

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset p_eqn.solver = _workspace(solvers.p_rgh.solver, _b(p_eqn))

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals = solver_variant(
        model, turbulenceModel, ∇p, ∇p_rgh, U_eqn, p_eqn,
        mdotf, rhoPhi, gh, ghf, phi_g, phi_gf, extra_models, mules, config;
        output=output, pref=pref,
        ncorrectors=ncorrectors, inner_loops=inner_loops)

    return residuals
end



function MULTIPHASE(
    model, turbulenceModel, ∇p, ∇p_rgh, U_eqn, p_eqn,
    mdotf, rhoPhi, gh, ghf, phi_g, phi_gf,
    extra_models, mules, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=3
    )

    (; alpha_prev, div_alpha, div_mdotf, alpha_fluxf,
       alphaf_upwind, alphaf_HO, phiLf, phiHf, phiAf,
       Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
       alphaMaxLocal, alphaMinLocal) = mules

    (; U, p) = model.momentum
    (; nu, nuf, rho, rhof, alpha, alphaf, p_rgh, p_rghf) = model.fluid
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware

    mesh      = model.domain
    mp_model  = model.fluid.model
    phases    = model.fluid.phases
    main      = model.fluid.volume_fraction
    secondary = 3 - main

    TF      = _get_float(mesh)
    TI      = _get_int(mesh)
    n_cells = length(mesh.cells)

    @info "Allocating working memory..."

    dt_cpu = zeros(TF, 1)
    copyto!(dt_cpu, runtime.dt)
    postprocess = convert_time_to_iterations(postprocess, model, dt_cpu[1], iterations)

    mueff = get_flux(U_eqn, 3)
    rDf   = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)
    nueff = FaceScalarField(mesh)

    outputWriter = initialise_writer(output, mesh)

    gradU  = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(mesh)
    S  = StrainRate(gradU, gradUT, U, Uf)

    # Aux fields for discretisation consistency
    ∇p_rghf_deconstructed = FaceScalarField(mesh)
    ∇p_rghf_reconstructed = VectorField(mesh)
    pressure_force_face   = FaceScalarField(mesh)

    rho1_val = phases[main].rho[1]
    rho2_val = phases[secondary].rho[1]
    mu1_val  = phases[main].mu[1]
    mu2_val  = phases[secondary].mu[1]

    rho1f = FaceScalarField(mesh); initialise!(rho1f, rho1_val)
    rho2f = FaceScalarField(mesh); initialise!(rho2f, rho2_val)
    mu1f  = FaceScalarField(mesh); initialise!(mu1f,  mu1_val)
    mu2f  = FaceScalarField(mesh); initialise!(mu2f,  mu2_val)

    ∇alpha  = Grad{schemes.alpha.gradient}(alpha)
    ∇alphaf = FaceVectorField(mesh)

    if typeof(mp_model) <: Mixture
        div_slip_momentum, = extra_models
        Sc_t     = 0.7
        C_alpha  = 0.0
        g_vec    = model.fluid.physics_properties.gravity.g
        diameter = mp_model.diameter
        tau_d    = (rho2_val * diameter^2) / (18.0 * mu1_val + eps())
        tau_d_field = ConstantScalar(tau_d)
        # Assumes constant values for now

        U_prev = VectorField(mesh)
        DUmDt  = VectorField(mesh)
        Ur     = VectorField(mesh)
        Urf    = FaceVectorField(mesh)
        ∇U     = Grad{schemes.U.gradient}(U)
    end

    if typeof(mp_model) <: VOF
        sigma   = mp_model.sigma
        C_alpha = mp_model.cAlpha

        nhatf_prep     = FaceVectorField(mesh)
        kappa          = ScalarField(mesh)
        kappaf         = FaceScalarField(mesh)
        grad_alpha_mag = ScalarField(mesh)
    else
        sigma  = zero(TF)
        kappaf = ConstantScalar(zero(TF))
    end

    phirf    = FaceScalarField(mesh)
    Urdotf   = FaceScalarField(mesh)

    Hv       = VectorField(mesh)
    rD       = ScalarField(mesh)
    rho_prev = ScalarField(mesh)
    prev     = KernelAbstractions.zeros(backend, TF, n_cells)

    R_ux    = ones(TF, iterations)
    R_uy    = ones(TF, iterations)
    R_uz    = ones(TF, iterations)
    R_p     = ones(TF, iterations)
    R_alpha = ones(TF, iterations)
    cellsCourant      = KernelAbstractions.zeros(backend, TF, n_cells)
    cellsAlphaCourant = KernelAbstractions.zeros(backend, TF, n_cells)

    time = zero(TF)
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    @. rhoPhi.values = mdotf.values * rhof.values
    update_nueff!(nueff, nuf, model.turbulence, config)
    @. mueff.values  = rhof.values * nueff.values

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Starting multiphase solver..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations

        copyto!(dt_cpu, config.runtime.dt)
        time += dt_cpu[1]

        @. rho_prev.values = rho.values

        if typeof(mp_model) <: Mixture
            @. U_prev.x.values = U.x.values
            @. U_prev.y.values = U.y.values
            @. U_prev.z.values = U.z.values

            grad!(∇U, Uf, U, boundaries.U, time, config)
        end

        if typeof(mp_model) <: Mixture
            grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
            limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)

            compute_DUmDt!(DUmDt, U, U_prev, ∇U, dt_cpu[1], config)
            compute_Ur!(Ur, alpha, rho, g_vec, DUmDt,
                        phases[main].rho, phases[secondary].rho, phases[main].mu,
                        diameter, tau_d_field, config)
            turbulent_dispersion!(Ur, alpha, ∇alpha, model.turbulence, Sc_t, config)
            
            interpolate_vanleer!(Urf, Ur, mdotf, config)
            zero_wall_drift_velocity!(Urf, config)
            face_dot_Sf!(Urdotf, Urf, config)
        end

        # Bounded alpha eqn. transport via MULES
        advance_alpha!(model, mp_model, ∇alpha, ∇alphaf, mdotf,
                       alpha_prev, alphaf_upwind, alphaf_HO, phirf, Urdotf, phiLf, phiHf, phiAf,
                       alpha_fluxf, div_alpha, div_mdotf,
                       Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                       alphaMaxLocal, alphaMinLocal, C_alpha, dt_cpu[1], time, config)
        ralpha = zero(TF)

        # Mixture property update from the new alpha
        update_mixture_properties!(model, alpha_fluxf, mdotf, rhoPhi, nueff, mueff,
                                   rho1_val, rho2_val, mu1_val, mu2_val, config)

        # Interface curvature for surface tension (VOF only)
        if typeof(mp_model) <: VOF
            update_curvature!(model, ∇alpha, ∇alphaf, nhatf_prep, kappa, kappaf,
                              grad_alpha_mag, time, config)
        end
        
        # Drift flux divergence (Mixture only)
        if typeof(mp_model) <: Mixture
            div_slip_outer!(div_slip_momentum, alphaf, rhof, rho1f, rho2f, Urf, config)
        end

        well_balanced_pressure_grad!(
            ∇p_rgh.result, pressure_force_face,
            p_rgh, rho, ghf, mesh, config;
            sigma=sigma, kappaf=kappaf, alpha=alpha)

        rx, ry, rz = solve_equation!(
            U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config; rho_prev=rho_prev, time=time)

        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)

        remove_pressure_source!(U_eqn, ∇p_rgh, config)

        rp = 0.0
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn, config)

            interpolate!(Uf, Hv, config)
            correct_boundaries!(Uf, Hv, boundaries.U, time, config)

            flux!(mdotf, Uf, config)

            phi_gf!(phi_gf, rho, ghf, rDf, model, config)

            if typeof(mp_model) <: VOF
                surface_tension_flux!(rDf, sigma, kappaf, alpha, phi_gf, config)
            end

            reconstruct!(phi_g, phi_gf, config)

            @. mdotf.values += phi_gf.values

            div!(divHv, mdotf, config)

            @. prev = p_rgh.values
            rp = solve_equation!(p_eqn, p_rgh, boundaries.p_rgh, solvers.p_rgh,
                                 config; ref=pref, time=time)

            grad!(∇p_rgh, p_rghf, p_rgh, boundaries.p_rgh, time, config)
            limit_gradient!(schemes.p_rgh.limiter, ∇p_rgh, p_rgh, config)

            correct_mass_flux_mp!(mdotf, p_eqn, config)

            pressure_grad!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, config)
            reconstruct!(∇p_rghf_reconstructed, ∇p_rghf_deconstructed, config)

            correct_velocity_rgh!(U, Hv, ∇p_rghf_reconstructed, rD, config)
        end

        @. p.values = p_rgh.values + (rho.values * gh.values)

        turbulence!(turbulenceModel, model, S, prev, time, config)
        update_nueff!(nueff, nuf, model.turbulence, config)
        @. mueff.values = rhof.values * nueff.values

        courant      = max_courant_number!(cellsCourant, model, config)
        alphaCourant = max_alpha_courant_number!(cellsAlphaCourant, alpha, mdotf, model, config, dt_cpu[1])
        update_dt!(config.runtime, courant, alphaCourant)

        R_ux[iteration]    = rx
        R_uy[iteration]    = ry
        R_uz[iteration]    = rz
        R_p[iteration]     = rp
        R_alpha[iteration] = ralpha

        ProgressMeter.next!(
            progress, showvalues = [
                (:dt, dt_cpu[1]),
                (:time, time),
                (:Courant, courant),
                (:AlphaCourant, alphaCourant),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p_rgh, R_p[iteration]),
                (:alpha, R_alpha[iteration]),
                turbulenceModel.state.residuals...
                ]
            )

        runtime_postprocessing!(postprocess,iteration,iterations,S,time,config)

        if iteration % write_interval + signbit(write_interval) == 0
            save_output(model, outputWriter, iteration, time, config)
            save_postprocessing(postprocess, iteration, time, mesh, outputWriter, config.boundaries)
        end
    end

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p, alpha=R_alpha)
end



function advance_alpha!(model, mp_model, ∇alpha, ∇alphaf, mdotf,
                        alpha_prev, alphaf_upwind, alphaf_HO, phirf, Urdotf, phiLf, phiHf, phiAf,
                        alpha_fluxf, div_alpha, div_mdotf,
                        Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                        alphaMaxLocal, alphaMinLocal, C_alpha, dt, time, config)
    (; alpha, alphaf) = model.fluid
    (; schemes, boundaries) = config
    mesh = model.domain

    @. alpha_prev.values = alpha.values

    grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
    limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)

    alpha_compression_flux!(mp_model, phirf, ∇alphaf, ∇alpha, mdotf, C_alpha, config)

    interpolate_upwind!(alphaf_upwind, alpha, mdotf, config)
    correct_boundaries!(alphaf_upwind, alpha, boundaries.alpha, time, config)
    interpolate_vanleer!(alphaf_HO, alpha, ∇alpha, mdotf, config)
    correct_boundaries!(alphaf_HO, alpha, boundaries.alpha, time, config)

    @. phiLf.values = mdotf.values * alphaf_upwind.values

    high_order_alpha_flux!(mp_model, phiHf, mdotf, alphaf_HO, alphaf_upwind, phirf, Urdotf)

    @. phiAf.values = phiHf.values - phiLf.values

    zero_boundary_faces!(phiAf, config)

    mules_limit!(mp_model,
                 phiAf, alpha_prev, phiLf,
                 Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                 alphaMaxLocal, alphaMinLocal,
                 dt, mesh, config)

    @. alpha_fluxf.values = phiLf.values + phiAf.values
    div!(div_alpha, alpha_fluxf, config)
    div!(div_mdotf, mdotf, config)
    @. alpha.values = alpha_prev.values -
        dt * (div_alpha.values - alpha_prev.values * div_mdotf.values)

    interpolate_vanleer!(alphaf, alpha, ∇alpha, mdotf, config)
    correct_boundaries!(alphaf, alpha, boundaries.alpha, time, config)
    return nothing
end

function alpha_compression_flux!(::VOF, phirf, ∇alphaf, ∇alpha, mdotf, C_alpha, config)
    interpolate!(∇alphaf, ∇alpha.result, config)
    compression_flux!(phirf, ∇alphaf, mdotf, C_alpha, config)
end
alpha_compression_flux!(::Mixture, phirf, ∇alphaf, ∇alpha, mdotf, C_alpha, config) = nothing

high_order_alpha_flux!(::VOF, phiHf, mdotf, alphaf_HO, alphaf_upwind, phirf, Urdotf) =
    @. phiHf.values = mdotf.values * alphaf_HO.values +
                        phirf.values * alphaf_HO.values * (1.0 - alphaf_HO.values)

high_order_alpha_flux!(::Mixture, phiHf, mdotf, alphaf_HO, alphaf_upwind, phirf, Urdotf) =
    @. phiHf.values = mdotf.values * alphaf_HO.values -
                        Urdotf.values * alphaf_upwind.values * (1.0 - alphaf_upwind.values)



function update_mixture_properties!(model, alpha_fluxf, mdotf, rhoPhi, nueff, mueff,
                                    rho1_val, rho2_val, mu1_val, mu2_val, config)
    (; rho, rhof, nu, nuf, alpha, alphaf) = model.fluid

    blend_properties!(rho,  alpha,  rho1_val, rho2_val)
    blend_properties!(rhof, alphaf, rho1_val, rho2_val)
    blend_mixture_nu!(nu,  alpha,  rho,  mu1_val, mu2_val)
    blend_mixture_nu!(nuf, alphaf, rhof, mu1_val, mu2_val)

    update_nueff!(nueff, nuf, model.turbulence, config)
    @. mueff.values  = rhof.values * nueff.values

    blend_rhoPhi!(model.fluid.model, rhoPhi, alpha_fluxf, mdotf, rhof, rho1_val, rho2_val)
    return nothing
end

blend_rhoPhi!(::Mixture, rhoPhi, alpha_fluxf, mdotf, rhof, rho1_val, rho2_val) =
    @. rhoPhi.values = mdotf.values * rhof.values

blend_rhoPhi!(::VOF, rhoPhi, alpha_fluxf, mdotf, rhof, rho1_val, rho2_val) =
    @. rhoPhi.values = alpha_fluxf.values * (rho1_val - rho2_val) + mdotf.values * rho2_val



function update_curvature!(model, ∇alpha, ∇alphaf, nhatf_prep, kappa, kappaf,
                           grad_alpha_mag, time, config)
    (; alpha, alphaf) = model.fluid
    (; schemes, boundaries) = config
    grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
    limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)
    interpolate!(∇alphaf, ∇alpha.result, config)
    nhat_prep!(nhatf_prep, alpha, ∇alphaf, config)
    div!(kappa, nhatf_prep, config)
    cell_grad_magnitude!(grad_alpha_mag, ∇alpha, config)
    interpolate_weighted!(kappaf, kappa, grad_alpha_mag, config)
    return nothing
end


"""
    blend_properties!(property_field, alpha_field, property_0, property_1)

Linearly blends a per-phase scalar property using the volume-fraction field:

    property_field = property_0 * alpha + property_1 * (1 - alpha)
"""
function blend_properties!(property_field, alpha_field, property_0, property_1)
    @. property_field.values = (property_0 * alpha_field.values) + (property_1 * (1.0 - alpha_field.values))
    nothing
end

"""
    blend_mixture_nu!(nu_field, alpha_field, rho_field, mu_0, mu_1)

Mixture kinematic viscosity built from dynamic viscosity field:

    nu = (mu_0 * alpha + mu_1 * (1 - alpha)) / rho_blend
"""
function blend_mixture_nu!(nu_field, alpha_field, rho_field, mu_0, mu_1)
    @. nu_field.values = (mu_0 * alpha_field.values + mu_1 * (1.0 - alpha_field.values)) / rho_field.values
    nothing
end


"""
    compute_gh!(gh, g, config)

Computes `g . x` at cell centres. Used for hydrostatic pressure reconstruction.
"""
function compute_gh!(gh, g, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = gh.mesh.cells

    ndrange = length(gh)
    kernel! = _compute_gh!(_setup(backend, workgroup, ndrange)...)
    kernel!(gh, g, cells)
end
@kernel inbounds=true function _compute_gh!(gh, g, cells)
    i = @index(Global)
    (; centre) = cells[i]
    gh[i] = (g ⋅ centre)
end


"""
    compute_ghf!(ghf, g, config)

Computes `g . x` at face centres.
"""
function compute_ghf!(ghf, g, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    faces = ghf.mesh.faces

    ndrange = length(ghf)
    kernel! = _compute_ghf!(_setup(backend, workgroup, ndrange)...)
    kernel!(ghf, g, faces)
end
@kernel inbounds=true function _compute_ghf!(ghf, g, faces)
    i = @index(Global)
    (; centre) = faces[i]
    ghf[i] = (g ⋅ centre)
end



"""
    phi_gf!(phi_gf, rho, ghf, rDf, model, config)

Builds the gravity (buoyancy) contribution to the face mass flux. On each
face computes

    phi_gf[f] = -ghf[f] · area · snGrad(rho) · rDf[f]

It is summed into `mdotf` before the pressure solve and reconstructed to
a cell vector (`phi_g`) for the velocity correction. A face with 
no density jump (single-phase region) contributes zero.
"""
function phi_gf!(phi_gf, rho, ghf, rDf, model, config)
    (; faces, cells, boundary_cellsID) = model.domain
    (; hardware) = config
    (; backend, workgroup) = hardware

    n_bfaces = length(boundary_cellsID)

    ndrange = length(faces)
    kernel! = _phi_gf!(_setup(backend, workgroup, ndrange)...)
    kernel!(phi_gf, rho, ghf, rDf, faces, cells, n_bfaces)
end
@kernel function _phi_gf!(phi_gf, rho, ghf, rDf, faces, cells, n_bfaces)
    fID = @index(Global)
    @inbounds begin
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        rho1 = rho[cID1]
        rho2 = rho[cID2]

        face_grad = area * (rho2 - rho1) / delta

        phi_gf[fID] = -ghf[fID] * face_grad * rDf[fID]
    end
end

"""
    pressure_grad!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, config)

Builds the face-normal pressure-gradient field used to correct the cell
velocity (Rhie-Chow consistent). On each face computes

    ∇p_rghf_deconstructed[f] = (phi_gf[f] - snGrad(p_rgh)·area·rDf[f]) / (rDf[f])

The result is a face scalar later reconstructed to a cell vector (`reconstruct!`) 
and fed to `correct_velocity_rgh!`.
"""
function pressure_grad!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    faces = ∇p_rghf_deconstructed.mesh.faces

    ndrange = length(∇p_rghf_deconstructed)
    kernel! = _pressure_grad!(_setup(backend, workgroup, ndrange)...)
    kernel!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, faces)
end
@kernel function _pressure_grad!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, faces)
    i = @index(Global)
    face = faces[i]
    (; area, normal, ownerCells, delta) = face

    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    p1 = p_rgh[cID1]
    p2 = p_rgh[cID2]
    face_grad = area * (p2 - p1) / delta

    ∇p_rghf_deconstructed[i] = (phi_gf[i] - (face_grad * rDf[i])) / (rDf[i] + eps())
end

"""
    correct_velocity_rgh!(U, Hv, ∇p, rD, config)

Slightly modified version of `correct_velocity!` for convenience.
"""
function correct_velocity_rgh!(U, Hv, ∇p, rD, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(U)
    kernel! = _correct_velocity_rgh!(_setup(backend, workgroup, ndrange)...)
    kernel!(U, Hv, ∇p, rD)
end
@kernel function _correct_velocity_rgh!(U, Hv, ∇p, rD)
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hvx, Hvy, Hvz = Hv.x, Hv.y, Hv.z
        dpdx, dpdy, dpdz = ∇p.x, ∇p.y, ∇p.z
        rDvalues = rD.values
    end

    @inbounds begin
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] + dpdx[i] * rDvalues_i
        Uy[i] = Hvy[i] + dpdy[i] * rDvalues_i
        Uz[i] = Hvz[i] + dpdz[i] * rDvalues_i
    end
end

"""
    reconstruct!(phi::VectorField, psif::FaceScalarField, config)

Least-squares reconstruction of a cell-centred vector field `phi` from a
face-normal scalar field `psif`. Required for discretisation consistency.

"""
function reconstruct!(phi::VectorField, psif::FaceScalarField, config)
    mesh = phi.mesh
    (; cells, cell_nsign, cell_faces, faces) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    F = _get_float(mesh)
    ndrange = length(cells)

    if typeof(mesh) <: Mesh2
        kernel! = _reconstruct_operation_2D!(_setup(backend, workgroup, ndrange)...)
    else
        kernel! = _reconstruct_operation_3D!(_setup(backend, workgroup, ndrange)...)
    end
    kernel!(cells, F, cell_faces, cell_nsign, faces, phi, psif)
end

@kernel function _reconstruct_operation_2D!(
    cells::AbstractArray{Cell{TF,SV,UR}}, F, cell_faces, cell_nsign, faces, phi, psif
) where {TF,SV,UR}
    i = @index(Global)
    @inbounds begin
        (; faces_range) = cells[i]

        m11 = zero(TF); m12 = zero(TF); m22 = zero(TF)
        b1  = zero(TF); b2  = zero(TF)

        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area, normal) = faces[fID]
            nx = normal[1]; ny = normal[2]

            m11 += area * nx * nx
            m12 += area * nx * ny
            m22 += area * ny * ny

            ssf = psif[fID]
            b1 += nx * ssf
            b2 += ny * ssf
        end

        det = m11*m22 - m12*m12

        is_invertible = abs(det) > eps(TF)
        invdet = is_invertible ? one(TF)/det : zero(TF)

        ux = ( m22*b1 - m12*b2) * invdet
        uy = (-m12*b1 + m11*b2) * invdet

        phi[i] = @SVector [ux, uy, zero(TF)]
    end
end

@kernel function _reconstruct_operation_3D!(
    cells::AbstractArray{Cell{TF,SV,UR}}, F, cell_faces, cell_nsign, faces, phi, psif
) where {TF,SV,UR}
    i = @index(Global)
    @inbounds begin
        (; faces_range) = cells[i]

        m11 = zero(TF); m12 = zero(TF); m13 = zero(TF)
                        m22 = zero(TF); m23 = zero(TF)
                                        m33 = zero(TF)
        b1 = zero(TF);  b2 = zero(TF);  b3 = zero(TF)

        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area, normal) = faces[fID]
            nx = normal[1]; ny = normal[2]; nz = normal[3]

            m11 += area * nx * nx
            m12 += area * nx * ny
            m13 += area * nx * nz
            m22 += area * ny * ny
            m23 += area * ny * nz
            m33 += area * nz * nz

            ssf = psif[fID]
            b1 += nx * ssf
            b2 += ny * ssf
            b3 += nz * ssf
        end

        A11 = m22*m33 - m23*m23
        A12 = m13*m23 - m12*m33
        A13 = m12*m23 - m13*m22

        det = m11*A11 + m12*A12 + m13*A13

        is_invertible = abs(det) > eps(TF)
        invdet = is_invertible ? one(TF)/det : zero(TF)

        ux = (A11*b1 + A12*b2 + A13*b3) * invdet
        uy = (A12*b1 + (m11*m33 - m13*m13)*b2 + (m13*m12 - m11*m23)*b3) * invdet
        uz = (A13*b1 + (m13*m12 - m11*m23)*b2 + (m11*m22 - m12*m12)*b3) * invdet

        phi[i] = @SVector [ux, uy, uz]
    end
end


# Removes boundary-related problems, required for mass conservation in some cases.
function zero_boundary_faces!(phif::FaceScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = phif.mesh
    nbfaces = length(mesh.boundary_cellsID)
    if nbfaces > 0
        ndrange = nbfaces
        kernel! = _mmp_zero_boundary_faces!(_setup(backend, workgroup, ndrange)...)
        kernel!(phif)
    end
end
@kernel inbounds=true function _mmp_zero_boundary_faces!(phif)
    i = @index(Global)
    phif[i] = zero(eltype(phif.values))
end


# Zero the drift velocity on boundary faces so no slip flux through walls
function zero_wall_drift_velocity!(Urf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    nbfaces = length(Urf.mesh.boundary_cellsID)

    if nbfaces > 0
        ndrange = nbfaces
        kernel! = _zero_wall_drift_velocity!(_setup(backend, workgroup, ndrange)...)
        kernel!(Urf)
    end
end
@kernel inbounds=true function _zero_wall_drift_velocity!(Urf)
    i = @index(Global)
    TF = eltype(Urf.x)
    Urf.x[i] = zero(TF)
    Urf.y[i] = zero(TF)
    Urf.z[i] = zero(TF)
end


"""
    mules_limit!(mp_model::AbstractMultiphaseModel,
                 phiAf, alpha_prev, phiLf,
                 Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                 alphaMaxLocal, alphaMinLocal,
                 dt, mesh, config)

Unified MULES (Zalesak FCT) flux-limiter for explicit alpha transport, shared by
both `VOF` and `Mixture` sub models. Limits the anti-diffusive
face flux `phiAf = phiHf - phiLf` so the explicit update
`α^{n+1} = α^n - dt/V · div(phiLf + phiAf)` stays bounded.

Logic:

  1. `mules_set_bounds!`: sets per-cell `alphaMax`, `alphaMin`
      Dispatched:
        - `VOF`     local neighbour extrema of `alpha_prev` between [0,1];
                    prevents isolated alpha=1 cells; preserves sharp interface.
        - `Mixture` hard `[0,1]` limit; must tolerate uniform alpha
                                without freezing the anti-diffusive drift flux.
  2. Build low-order α*, P+-, Q+-
  3. Ratios: R+- = clamp(Q+-/P+-, 0, 1).
  4. Per-face apply lambda_f (scales phiAf).

Boundary faces are left unlimited (lambda = 1).
"""
function mules_limit!(mp_model::AbstractMultiphaseModel,
                      phiAf, alpha_prev, phiLf,
                      Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                      alphaMaxLocal, alphaMinLocal,
                      dt, mesh, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; cells, cell_nsign, cell_faces, faces) = mesh

    n_cells  = length(cells)
    n_faces  = length(faces)
    nbfaces  = length(mesh.boundary_cellsID)

    fill!(Pplus.values,  0)
    fill!(Pminus.values, 0)

    # Required to prevent isolated alpha=1 cells
    mules_set_bounds!(mp_model, alphaMaxLocal, alphaMinLocal, alpha_prev, mesh, config)

    # Numerical consistency
    ndrange = n_cells
    kernel! = _mmp_mules_cell_accum!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, cell_faces, cell_nsign, faces,
            alpha_prev, phiLf, phiAf,
            Pplus, Pminus, Qplus, Qminus,
            alphaMaxLocal, alphaMinLocal, dt)

    ndrange = n_cells
    kernel! = _mmp_mules_ratios!(_setup(backend, workgroup, ndrange)...)
    kernel!(Pplus, Pminus, Qplus, Qminus, Rplus, Rminus)

    # Core MULES operation (compute phi contribution bounded by lambda)
    ndrange = n_faces
    kernel! = _mmp_mules_apply!(_setup(backend, workgroup, ndrange)...)
    kernel!(phiAf, faces, Rplus, Rminus, nbfaces)
end

function mules_set_bounds!(::Mixture, alphaMaxLocal, alphaMinLocal, alpha_prev, mesh, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(mesh.cells)
    kernel! = _mmp_mules_hard_bounds!(_setup(backend, workgroup, ndrange)...)
    kernel!(alphaMaxLocal, alphaMinLocal)
end

function mules_set_bounds!(::VOF, alphaMaxLocal, alphaMinLocal, alpha_prev, mesh, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; cells, cell_faces, faces) = mesh

    ndrange = length(cells)
    kernel! = _mmp_mules_stencil_bounds!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, cell_faces, faces, alpha_prev, alphaMaxLocal, alphaMinLocal)
end

@kernel inbounds=true function _mmp_mules_hard_bounds!(alphaMaxLocal, alphaMinLocal)
    i = @index(Global)
    TF = eltype(alphaMaxLocal.values)
    alphaMaxLocal[i] = one(TF)
    alphaMinLocal[i] = zero(TF)
end

@kernel inbounds=true function _mmp_mules_stencil_bounds!(
    cells::AbstractArray{Cell{TF,SV,UR}}, cell_faces, faces,
    alpha_prev, alphaMaxLocal, alphaMinLocal
) where {TF,SV,UR}
    i = @index(Global)
    (; faces_range) = cells[i]

    aMax = alpha_prev[i]
    aMin = alpha_prev[i]

    for fi in faces_range
        fID = cell_faces[fi]
        oc = faces[fID].ownerCells

        j = ifelse(oc[1] == i, oc[2], oc[1])

        aj = alpha_prev[j]
        aMax = max(aMax, aj)
        aMin = min(aMin, aj)
    end

    alphaMaxLocal[i] = min(one(TF),  aMax)
    alphaMinLocal[i] = max(zero(TF), aMin)
end

@kernel inbounds=true function _mmp_mules_cell_accum!(
    cells::AbstractArray{Cell{TF,SV,UR}}, cell_faces, cell_nsign, faces,
    alpha_prev, phiLf, phiAf, Pplus, Pminus, Qplus, Qminus,
    alphaMaxLocal, alphaMinLocal, dt
) where {TF,SV,UR}
    i = @index(Global)
    (; volume, faces_range) = cells[i]

    sum_L = zero(TF)
    sum_A_pos = zero(TF)
    sum_A_neg = zero(TF)

    for fi in faces_range
        fID   = cell_faces[fi]
        nsign = cell_nsign[fi]

        fL = phiLf[fID] * nsign
        fA = phiAf[fID] * nsign
        sum_L += fL

        if fA < zero(TF)
            sum_A_pos += -fA
        else
            sum_A_neg += fA
        end
    end

    alpha_star = alpha_prev[i] - dt / volume * sum_L

    Pplus[i]  = sum_A_pos
    Pminus[i] = sum_A_neg

    Qplus[i]  = max(zero(TF), (alphaMaxLocal[i] - alpha_star)) * volume / dt
    Qminus[i] = max(zero(TF), (alpha_star - alphaMinLocal[i])) * volume / dt
end

@kernel inbounds=true function _mmp_mules_ratios!(Pplus, Pminus, Qplus, Qminus, Rplus, Rminus)
    i = @index(Global)
    TF = eltype(Pplus.values)

    Pp = Pplus[i]; Pm = Pminus[i]
    Qp = Qplus[i]; Qm = Qminus[i]

    Rplus[i]  = Pp > eps(TF) ? clamp(Qp / Pp, zero(TF), one(TF)) : one(TF)
    Rminus[i] = Pm > eps(TF) ? clamp(Qm / Pm, zero(TF), one(TF)) : one(TF)
end

@kernel inbounds=true function _mmp_mules_apply!(phiAf, faces, Rplus, Rminus, nbfaces)
    i = @index(Global)
    TF = eltype(phiAf.values)
    if i > nbfaces
        face = faces[i]
        (; ownerCells) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
    
        fA = phiAf[i]

        lambda = if fA > zero(TF)
            min(Rplus[cID2], Rminus[cID1])
        elseif fA < zero(TF)
            min(Rplus[cID1], Rminus[cID2])
        else
            one(TF)
        end

        phiAf[i] = lambda * fA
    end
end


"""
    compression_flux!(phirf, ∇alphaf, mdotf, C_alpha, config)

Anti-diffusive compression face flux:

    phir_f · Sf = min(Cα · |phi_f|/|Sf|, Phi_max) · (nnhatf · Sf)
"""
function compression_flux!(phirf, ∇alphaf, mdotf, C_alpha, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh  = phirf.mesh
    faces = mesh.faces
    TF = _get_float(mesh)

    # Phi max limits compression flux for numerical stability, compute it separately:
    phi_over_S_buf = similar(mdotf.values)
    ndrange = length(faces)
    kernel! = _fill_phi_over_S!(_setup(backend, workgroup, ndrange)...)
    kernel!(phi_over_S_buf, mdotf, faces)
    phimax = maximum(phi_over_S_buf)

    ndrange = length(faces)
    kernel! = _compression_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(phirf, ∇alphaf, mdotf, faces, TF(C_alpha), TF(phimax))
end

@kernel inbounds=true function _fill_phi_over_S!(buf, mdotf, faces)
    i = @index(Global)
    TF = eltype(buf)
    area = faces[i].area
    buf[i] = abs(mdotf[i]) / (area + eps(TF))
end

@kernel inbounds=true function _compression_flux!(phirf, ∇alphaf, mdotf, faces, C_alpha, phimax)
    i = @index(Global)
    face = faces[i]
    (; area, normal, delta) = face
    TF = eltype(phirf.values)

    Sf = area * normal
    grad_alpha     = ∇alphaf[i]
    grad_alpha_mag = norm(grad_alpha)

    noise_floor = TF(1.0e-8) / delta

    if grad_alpha_mag > noise_floor
        nhat = grad_alpha / grad_alpha_mag
    else
        nhat = zero(grad_alpha)
    end

    phi_over_S = abs(mdotf[i]) / (area + eps(TF))
    compr_speed = min(C_alpha * phi_over_S, phimax)

    phirf[i] = compr_speed * (nhat ⋅ Sf)
end

"""
    cell_grad_magnitude!(mag_field, grad, config)

Cell-centred `|∇alpha|` from a gradient field, used as the |∇alpha|-weighted face
interpolation weight for kappa_f so that near wall cells with small |∇alpha| don't
ruin the surface-tension force.
"""
function cell_grad_magnitude!(mag_field, grad, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = mag_field.mesh
    ndrange = length(mesh.cells)
    kernel! = _cell_grad_magnitude!(_setup(backend, workgroup, ndrange)...)
    kernel!(mag_field, grad.result)
end

@kernel inbounds=true function _cell_grad_magnitude!(mag_field, grad_result)
    i = @index(Global)
    mag_field.values[i] = norm(grad_result[i])
end

"""
    interpolate_weighted!(phif, phi, weight_field, config)

Weighted face interpolation: `phif[f] = (phi_c1·w_c1 + phi_c2·w_c2) /
(w_c1 + w_c2)`. With `w = |∇alpha|` it preserves kappa_f at interfaces and zeros
it in bulk where kappa is noisy.
"""
function interpolate_weighted!(phif::FaceScalarField, phi::ScalarField,
                                 weight_field::ScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = phif.mesh
    (; faces) = mesh

    ndrange = length(faces)
    kernel! = _interpolate_weighted!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, phi, weight_field, faces)
end

@kernel inbounds=true function _interpolate_weighted!(phif, phi, w, faces)
    i = @index(Global)
    face = faces[i]
    (; ownerCells) = face
    c1 = ownerCells[1]
    c2 = ownerCells[2]
    TF = eltype(phif.values)

    w1 = w.values[c1]
    w2 = w.values[c2]
    denom = w1 + w2 + eps(TF)
    phif[i] = (phi.values[c1] * w1 + phi.values[c2] * w2) / denom
end

"""
    nhat_prep!(nhatf_prep, alpha, ∇alphaf, config)

Builds a face-normal unit vector field from the face-interpolated alpha grad.
"""
function nhat_prep!(nhatf_prep, alpha, ∇alphaf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = alpha.mesh
    faces = mesh.faces
    nbfaces = length(mesh.boundary_cellsID)
    nfaces  = length(faces)

    if nbfaces > 0
        kernel! = _nhat_zero_bfaces!(_setup(backend, workgroup, nbfaces)...)
        kernel!(nhatf_prep)
    end

    ninternal = nfaces - nbfaces
    if ninternal > 0
        kernel! = _nhat_normalise_ifaces!(_setup(backend, workgroup, ninternal)...)
        kernel!(nhatf_prep, faces, ∇alphaf, nbfaces)
    end
end

@kernel inbounds=true function _nhat_zero_bfaces!(nhatf_prep)
    i = @index(Global)
    nhatf_prep[i] = SVector(0.0, 0.0, 0.0)
end

@kernel inbounds=true function _nhat_normalise_ifaces!(nhatf_prep, faces, ∇alphaf_, nbfaces)
    i = @index(Global)
    fID = i + nbfaces
    face = faces[fID]
    (; delta) = face

    grad_alpha     = ∇alphaf_[fID]
    grad_alpha_mag = norm(grad_alpha)

    noise_floor = 1e-8 / delta
    if grad_alpha_mag < noise_floor
        nhatf_prep[fID] = SVector(0.0, 0.0, 0.0)
    else
        nhatf_prep[fID] = grad_alpha / grad_alpha_mag
    end
end

"""
    surface_tension_flux!(rDf, sigma, kappaf, alpha, phi_gf, config)

CSF surface-tension contribution to the face flux:
`phi_gf -= σ · κf · (∇αf · Sf) · rDf`.
"""
function surface_tension_flux!(rDf, sigma, kappaf, alpha, phi_gf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    faces = phi_gf.mesh.faces

    ndrange = length(phi_gf)
    kernel! = _surface_tension_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(rDf, sigma, kappaf, alpha, phi_gf, faces)
end

@kernel inbounds=true function _surface_tension_flux!(rDf, sigma, kappaf, alpha, phi_gf, faces)
    i = @index(Global)
    face = faces[i]
    (; area, normal, ownerCells, delta) = face
    Sf = area * normal

    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    alpha1 = alpha[cID1]
    alpha2 = alpha[cID2]

    ∇alphaf_vec = normal * ((alpha2 - alpha1) / delta)

    phi_gf[i] -= sigma * kappaf[i] * (∇alphaf_vec ⋅ Sf) * rDf[i]
end

"""
    well_balanced_pressure_grad!(grad_field, face_buf, p_rgh, rho, ghf,
                                  mesh, config; sigma=0, kappaf=nothing,
                                  alpha=nothing)

Overrides `grad_field.values` with a face-snGrad reconstruction of the predictor body force terms:

    face_buf[f] = area_f · ( snGrad(p_rgh) + ghf · snGrad(rho) - sigma·kappaf · snGrad(alpha) )

*Important for stability.
"""
function well_balanced_pressure_grad!(
    grad_field, face_buf, p_rgh, rho, ghf, mesh, config;
    sigma=zero(eltype(p_rgh.values)),
    kappaf, alpha,
)
    (; hardware) = config
    (; backend, workgroup) = hardware
    faces = mesh.faces

    ndrange = length(faces)
    kernel! = _well_balanced_pressure_face!(_setup(backend, workgroup, ndrange)...)
    kernel!(face_buf, p_rgh, rho, alpha, ghf, kappaf, sigma, faces)

    reconstruct!(grad_field, face_buf, config)
end

@kernel inbounds=true function _well_balanced_pressure_face!(
    face_buf, p_rgh, rho, alpha, ghf, kappaf, sigma, faces
)
    i = @index(Global)
    face = faces[i]
    (; area, ownerCells, delta) = face
    c1 = ownerCells[1]
    c2 = ownerCells[2]

    snGrad_p   = (p_rgh[c2] - p_rgh[c1]) / delta
    snGrad_rho = (rho[c2]   - rho[c1])   / delta
    snGrad_a   = (alpha[c2] - alpha[c1]) / delta

    face_buf[i] = area * (snGrad_p + ghf[i] * snGrad_rho - sigma * kappaf[i] * snGrad_a)
end


function correct_mass_flux_mp!(mdotf, p_eqn, config; time=nothing)
    # sngrad = FaceScalarField(mesh)
    (; faces, cells, boundary_cellsID) = mdotf.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    p = p_eqn.model.terms[1].phi
    A = _A(p_eqn)
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces # length(n_ifaces) was a BUG! should be n_ifaces only!!!!
    kernel! = _correct_mass_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(mdotf, p, nzval, colval, rowptr, faces, cells, n_bfaces)
    KernelAbstractions.synchronize(backend)

    BCs = config.boundaries.p_rgh # this line had to be changed from ".p"
    for BC ∈ BCs
        correct_mass_periodic(
            BC, mdotf, p, nzval, colval, rowptr, cells, faces, backend, workgroup)
        KernelAbstractions.synchronize(backend)
    end

    correct_boundary_mass_flux!(mdotf, p_eqn, BCs, time, config)
end




function compute_DUmDt!(DUmDt, U, U_prev, gradU, dt, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(DUmDt)
    kernel! = _compute_DUmDt!(_setup(backend, workgroup, ndrange)...)
    kernel!(DUmDt, U, U_prev, gradU.result, dt)
end

@kernel inbounds=true function _compute_DUmDt!(DUmDt, U, U_prev, gradU_result, dt)
    i = @index(Global)
    TF = eltype(U.x)

    dUdt = (U[i] - U_prev[i]) / dt

    ux = U.x[i]
    uy = U.y[i]
    uz = U.z[i]

    dudx = gradU_result.xx[i]
    dudy = gradU_result.xy[i]
    dudz = gradU_result.xz[i]

    dvdx = gradU_result.yx[i]
    dvdy = gradU_result.yy[i]
    dvdz = gradU_result.yz[i]

    dwdx = gradU_result.zx[i]
    dwdy = gradU_result.zy[i]
    dwdz = gradU_result.zz[i]

    conv_x = ux*dudx + uy*dudy + uz*dudz
    conv_y = ux*dvdx + uy*dvdy + uz*dvdz
    conv_z = ux*dwdx + uy*dwdy + uz*dwdz

    DUmDt[i] = dUdt + @SVector [conv_x, conv_y, conv_z]
end

function compute_Ur!(Ur, alpha, rho, g, DUmDt, rho1, rho2, mu1, d, tau_d, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(Ur)
    kernel! = _compute_Ur!(_setup(backend, workgroup, ndrange)...)
    kernel!(Ur, alpha, rho, g, DUmDt, rho1, rho2, mu1, d, tau_d)
end

@kernel inbounds=true function _compute_Ur!(Ur, alpha, rho, g, DUmDt, rho1, rho2, mu1, d, tau_d)
    i = @index(Global)
    TF = eltype(rho.values)

    rho_m = rho[i]
    rho_c = rho1[i]
    rho_d = rho2[i]
    mu_c  = mu1[i]
    tau   = tau_d[i]

    Ur_mag = norm(Ur[i])
    Re_p   = rho_c * Ur_mag * d / (mu_c + eps(TF))

    f_drag = ifelse(
        Re_p < TF(1000),
        one(TF) + TF(0.15) * Re_p^TF(0.687),
        TF(0.0183) * Re_p
    )
    f_drag = max(f_drag, one(TF))

    a_eff    = g - DUmDt[i]
    buoyancy = (rho_d - rho_m) / (rho_d + eps(TF))

    U_dm = (tau / (f_drag + eps(TF))) * buoyancy * a_eff

    alpha_c = max(alpha[i], TF(1e-3))
    Ur[i] = U_dm / alpha_c
end

turbulent_dispersion!(Ur, alpha, ∇alpha, turbulence::Laminar, Sc_t, config) = nothing

function turbulent_dispersion!(Ur, alpha, ∇alpha, turbulence, Sc_t, config)

    if !hasproperty(turbulence, :nut)
        return nothing
    end

    (; hardware) = config
    (; backend, workgroup) = hardware
    nut = turbulence.nut

    ndrange = length(Ur)
    kernel! = _turbulent_dispersion!(_setup(backend, workgroup, ndrange)...)
    kernel!(Ur, alpha, ∇alpha.result, nut, Sc_t)
end

@kernel inbounds=true function _turbulent_dispersion!(Ur, alpha, gradA, nut, Sc_t)
    i = @index(Global)
    TF = eltype(alpha.values)

    alpha_c = alpha[i]
    alpha_c_safe = max(alpha_c, TF(1e-3))
    alpha_d_safe = max(one(TF) - alpha_c, TF(1e-3))

    D_t   = nut[i] / TF(Sc_t)
    denom = alpha_c_safe * alpha_d_safe + eps(TF)
    coef  = D_t / denom

    gx = gradA.x[i]
    gy = gradA.y[i]
    gz = gradA.z[i]

    Ur[i] = Ur[i] + @SVector [coef*gx, coef*gy, coef*gz]
end

function face_dot_Sf!(phidotf, phif, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh  = phidotf.mesh
    faces = mesh.faces

    ndrange = length(faces)
    kernel! = _face_dot_Sf!(_setup(backend, workgroup, ndrange)...)
    kernel!(phidotf, phif, faces)
end

@kernel inbounds=true function _face_dot_Sf!(phidotf, phif, faces)
    i = @index(Global)
    (; area, normal) = faces[i]
    phidotf[i] = area * (phif.x[i]*normal[1] + phif.y[i]*normal[2] + phif.z[i]*normal[3])
end


@inline function _slip_coeff(alphaf, rhof, rho1f, rho2f, i, TF)
    af = alphaf[i]
    (af * (one(TF) - af) * (rho1f[i] + rho2f[i])) / (rhof[i] + eps(TF))
end

function div_slip_outer!(vector::VectorField, alphaf, rhof, rho1f, rho2f, Urf, config)
    mesh = vector.mesh
    (; cells, cell_nsign, cell_faces, faces) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(cells)
    kernel! = div_slip_outer_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, cell_faces, cell_nsign, faces, vector, alphaf, rhof, rho1f, rho2f, Urf)

    nbfaces = length(mesh.boundary_cellsID)
    ndrange = nbfaces
    kernel! = div_slip_outer_boundary_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(faces, cells, vector, alphaf, rhof, rho1f, rho2f, Urf)
end

@kernel inbounds=true function div_slip_outer_kernel!(cells::AbstractArray{Cell{TF,SV,UR}}, cell_faces, cell_nsign, faces, vector, alphaf, rhof, rho1f, rho2f, Urf) where {TF,SV,UR}
    i = @index(Global)

    @inbounds begin
        (; volume, faces_range) = cells[i]

        reduction_x = zero(TF)
        reduction_y = zero(TF)
        reduction_z = zero(TF)

        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, normal) = faces[fID]

            ux = Urf.x[fID]
            uy = Urf.y[fID]
            uz = Urf.z[fID]

            coeff = _slip_coeff(alphaf, rhof, rho1f, rho2f, fID, TF)
            w = coeff * (ux*normal[1] + uy*normal[2] + uz*normal[3]) * area * nsign

            reduction_x += w * ux
            reduction_y += w * uy
            reduction_z += w * uz
        end

        vector.x[i] = reduction_x / volume
        vector.y[i] = reduction_y / volume
        vector.z[i] = reduction_z / volume
    end
end

@kernel function div_slip_outer_boundary_kernel!(faces, cells, vector, alphaf, rhof, rho1f, rho2f, Urf)
    i = @index(Global)

    @inbounds begin
        TF = eltype(Urf.x)
        cID = faces[i].ownerCells[1]
        volume = cells[cID].volume
        (; area, normal) = faces[i]

        ux = Urf.x[i]
        uy = Urf.y[i]
        uz = Urf.z[i]

        coeff = _slip_coeff(alphaf, rhof, rho1f, rho2f, i, TF)
        w = coeff * (ux*normal[1] + uy*normal[2] + uz*normal[3]) * area / volume

        Atomix.@atomic vector.x.values[cID] += w * ux
        Atomix.@atomic vector.y.values[cID] += w * uy
        Atomix.@atomic vector.z.values[cID] += w * uz
    end
end