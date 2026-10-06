export csimple!

"""
    csimple!(
        model_in, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0, progress=true
    )

Compressible variant of the SIMPLE algorithm with a sensible enthalpy transport equation for the energy. 

# Input arguments

- `model` reference to a `Physics` model defined by the user.
- `config` Configuration structure defined by the user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only (default = `nothing`)
- `ncorrectors` number of non-orthogonality correction loops (default = `0`)
- `inner_loops` number to inner loops used in transient solver based on PISO algorithm (default = `0`)
- `transpose_stress` include the explicit viscous stress ∇·(μ_eff dev2((∇U)ᵀ)) in the momentum equation (default = `true`)

# Output

- `Ux` Vector of x-velocity residuals for each iteration.
- `Uy` Vector of y-velocity residuals for each iteration.
- `Uz` Vector of y-velocity residuals for each iteration.
- `p` Vector of pressure residuals for each iteration.
- `e` Vector of energy residuals for each iteration.

"""
function csimple!(
    model, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0, progress=true, transpose_stress=true
    )
    check_distributed_support(:CSIMPLE, model)

    residuals = setup_compressible_solvers(
        CSIMPLE, model, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops, progress=progress,
        transpose_stress=transpose_stress
        )
    return residuals
end

# Setup for all compressible algorithms
function setup_compressible_solvers(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0, progress=true,
    transpose_stress=true
    ) 

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    # model = adapt(hardware.backend, model_in)
    (; U, p, Uf, pf) = model.momentum
    (; rho) = model.fluid
    mesh = model.domain

    @info "Pre-allocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rhorDf = FaceScalarField(mesh)
    initialise!(rhorDf, 1.0)
    mueff = FaceScalarField(mesh)
    mueffgradUt = stress_source(mesh) # ∇·(μ_eff dev2((∇U)ᵀ)), stays zero if transpose_stress=false
    divHv = ScalarField(mesh)

    @info "Defining models..."

    U_eqn = (
        Time{schemes.U.time}(rho, U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(mueff, U) 
        == 
        - Source(∇p.result)
        + Source(mueffgradUt)
    ) → VectorEquation(U, boundaries.U)

    if typeof(model.fluid) <: WeaklyCompressible

        p_eqn = (
            - Laplacian{schemes.p.laplacian}(rhorDf, p) == - Source(divHv)
        ) → ScalarEquation(p, boundaries.p)

    elseif typeof(model.fluid) <: Compressible

        pconv = FaceScalarField(mesh)
        p_eqn = (
            - Laplacian{schemes.p.laplacian}(rhorDf, p) 
            + Divergence{schemes.p.divergence}(pconv, p) 
            == 
            - Source(divHv)
        ) → ScalarEquation(p, boundaries.p)

    end

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p.preconditioner, p_eqn)

    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()), _index_type(_A(U_eqn)))
    @reset p_eqn.solver = _workspace(solvers.p.solver, _b(p_eqn), _index_type(_A(p_eqn)))
  
    @info "Initialising energy model..."
    energyModel = initialise(model.energy, model, mdotf, rho, p_eqn, config)

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config;
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops, progress=progress,
        transpose_stress=transpose_stress)

    return residuals    
end # end function

function CSIMPLE(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config ; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0, progress=true,
    transpose_stress=true
    )
    
    # Extract model variables and configuration
    (; U, p, Uf, pf) = model.momentum
    (; nu, nuf, rho, rhof) = model.fluid
    (; nut) = model.turbulence

    mesh = model.domain
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware
    
    dt_cpu = zeros(_get_float(mesh), 1)
    copyto!(dt_cpu, config.runtime.dt)
    
    postprocess = convert_time_to_iterations(postprocess,model,dt_cpu[1],iterations)
    mdotf = get_flux(U_eqn, 2)
    mueff = get_flux(U_eqn, 3)
    mueffgradUt = get_source(U_eqn, 2)
    rhorDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)

    pconv = nothing # assign to variable to function scope
    if typeof(model.fluid) <: Compressible
        pconv = get_flux(p_eqn, 2)
    end

    outputWriter = initialise_writer(output, model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT, U, Uf)

    n_cells = length(mesh.cells)
    nueff = FaceScalarField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    Psi = ScalarField(mesh)
    Psif = FaceScalarField(mesh)

    τT_fluxes = stress_fluxes(mesh, transpose_stress)
    nonorthogonal_flux = ncorrectors > 0 ? FaceScalarField(mesh) : nothing

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = KernelAbstractions.zeros(backend, TF, n_cells) 
    p_boundary_reference = similar(prev)

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    R_e = ones(TF, iterations)
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    grad!(∇p, pf, p, boundaries.p, time, config)
    thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);
    @. rho.values = Psi.values * p.values
    @. rhof.values = Psif.values * pf.values
    flux!(mdotf, Uf, rhof, config)
    update_viscosity!(model.fluid, model.energy, config)
    update_nueff!(nueff, nuf, model.turbulence, config)
    @. mueff.values = nueff.values * rhof.values
    grad!(gradU, Uf, U, boundaries.U, time, config) # for the stress term of the first iteration
    limit_gradient!(schemes.U.limiter, gradU, U, config)


    @info "Starting CSIMPLE loops..."

    bar = _progress_bar(iterations, progress)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iteration ∈ 1:iterations
        time = iteration

        # gradU and mueff of the current velocity (updated by turbulence! below)
        transpose_stress!(mueffgradUt, τT_fluxes, mueff, gradU, boundaries.U, config)

        # Store previous values for next time step energy source terms
        @. model.energy.prevRhoK = rho.values*0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
        @. model.energy.prevP = p.values

        # Set up and solve momentum equations
        rx, ry, rz = solve_equation!(
            U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config; rho_prev=rho
            )

        # Solve energy equation and update thermo properties
        energy!(energyModel, model, mdotf, ∇p, gradU, mueff, time, dt_cpu[1], config)
        thermo_Psi!(model, Psi)
        thermo_Psi!(model, Psif, config)

        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rhorDf, rD, config)
        correct_interpolation_periodic(rhorDf, rD, boundaries.U, config)
        @. rhorDf.values *= rhof.values

        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        
        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, boundaries.U, time, config)

        if typeof(model.fluid) <: Compressible
            flux!(pconv, Uf, config)
            @. pconv.values *= Psif.values

            flux!(mdotf, Uf, config)
            @. mdotf.values *= rhof.values
            interpolate!(pf, p, config)
            correct_boundaries!(pf, p, boundaries.p, time, config)
            @. mdotf.values -= mdotf.values*Psif.values*pf.values/rhof.values
            div!(divHv, mdotf, config)

        elseif typeof(model.fluid) <: WeaklyCompressible
            flux!(mdotf, Uf, config)
            @. mdotf.values *= rhof.values
            div!(divHv, mdotf, config)
        end
        
        # Pressure calculations
        rp = 0.0
        @. prev = p.values
        @. p_boundary_reference = p.values
        if typeof(model.fluid) <: Compressible
            rp = solve_equation!(
                p_eqn, p, boundaries.p, solvers.p, config; 
                ref=nothing)
        elseif typeof(model.fluid) <: WeaklyCompressible
            rp = solve_equation!(p_eqn, p, boundaries.p, solvers.p, config; ref=nothing)
        end

        # non-orthogonal correction
        for i ∈ 1:ncorrectors
            grad!(∇p, pf, p, boundaries.p, time, config)
            limit_gradient!(schemes.p.limiter, ∇p, p, config)
            @. p_boundary_reference = p.values
            discretise!(p_eqn, p, config)
            apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time, config)
            setReference!(p_eqn, pref, 1, config)
            nonorthogonal_face_correction(
                p_eqn, ∇p, rhorDf, config; correction=nonorthogonal_flux)
            update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            rp = solve_system!(p_eqn, solvers.p, p, nothing, config)
        end

        if !isnothing(solvers.p.limit)
            pmin = solvers.p.limit[1]; pmax = solvers.p.limit[2]
            clamp!(p.values, pmin, pmax)
        end

        # Correct pressure-dependent fluxes before under-relaxing cell pressure.
        grad!(∇p, pf, p, boundaries.p, time, config)
        limit_gradient!(schemes.p.limiter, ∇p, p, config)
        if typeof(model.fluid) <: Compressible
            @. mdotf.values += pconv.values*(pf.values)
        end
        correct_mass_flux!(
            mdotf, p_eqn, config;
            previous=p_boundary_reference, time=time,
            nonorthogonal=nonorthogonal_flux)

        explicit_relaxation!(p, prev, solvers.p.relax, config)
        grad!(∇p, pf, p, boundaries.p, time, config)
        limit_gradient!(schemes.p.limiter, ∇p, p, config)
        correct_velocity!(U, Hv, ∇p, rD, config)
        # interpolate!(Uf, U, config) # Careful: reusing Uf for interpolation
        # correct_boundaries!(Uf, U, boundaries.U, time, config)
        
        # Perform turbulence calculations and update eddy viscosity
        turbulence!(turbulenceModel, model, S, prev, time, config)
        update_viscosity!(model.fluid, model.energy, config)
        update_nueff!(nueff, nuf, model.turbulence, config)

        if typeof(model.fluid) <: WeaklyCompressible
            rhorelax = solvers.p.relax
            @. rho.values = rho.values * (1-rhorelax) + Psi.values * p.values * rhorelax
            @. rhof.values = rhof.values * (1-rhorelax) + Psif.values * pf.values * rhorelax
        else
            @. rho.values = Psi.values * p.values
            @. rhof.values = Psif.values * pf.values
        end

        # update turbulent dynamic viscosity
        @. mueff.values = rhof.values*nueff.values
        if model.turbulence isa Laminar 
            @. model.energy.mueff_cell.values = rho.values*nu.values 
        else
            @. model.energy.mueff_cell.values = rho.values*(nu.values + nut.values)
        end



        # stor residuals and check for convergence
        R_ux[iteration] = rx
        R_uy[iteration] = ry
        R_uz[iteration] = rz
        R_p[iteration] = rp

        Uz_convergence = true
        if _base_mesh(mesh) isa Mesh3
            Uz_convergence = rz <= solvers.U.convergence
        end

        if (R_ux[iteration] <= solvers.U.convergence && 
            R_uy[iteration] <= solvers.U.convergence && 
            Uz_convergence &&
            R_p[iteration] <= solvers.p.convergence &&
            turbulenceModel.state.converged)

            isnothing(bar) || (bar.n = iteration; finish!(bar))
            @info "Simulation converged in $iteration iterations!"
            if !signbit(write_interval)
                save_output(model, outputWriter, iteration, time, config)
            end
            break
        end

        isnothing(bar) || ProgressMeter.next!(
            bar, showvalues = [
                (:iter,iteration),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p, R_p[iteration]),
                turbulenceModel.state.residuals...,
                energyModel.state.residuals
                ]
            )
        runtime_postprocessing!(postprocess,iteration,iterations,S,time,config)
        if iteration%write_interval + signbit(write_interval) == 0      
            save_output(model, outputWriter, iteration, time, config)
            save_postprocessing(
                postprocess,iteration,time,mesh,outputWriter,config.boundaries)
        end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p, e=R_e)
end
