export csimple!

"""
    csimple!(
        model_in, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

Compressible variant of the SIMPLE algorithm with a sensible enthalpy transport equation for the energy. 

# Input arguments

- `model` reference to a `Physics` model defined by the user.
- `config` Configuration structure defined by the user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only (default = `nothing`)
- `ncorrectors` number of non-orthogonality correction loops (default = `0`)
- `inner_loops` number to inner loops used in transient solver based on PISO algorithm (default = `0`)

# Output

- `Ux` Vector of x-velocity residuals for each iteration.
- `Uy` Vector of y-velocity residuals for each iteration.
- `Uz` Vector of y-velocity residuals for each iteration.
- `p` Vector of pressure residuals for each iteration.
- `e` Vector of energy residuals for each iteration.

"""
function csimple!(model, config; output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0) 

    residuals = setup_compressible_solvers(
        CSIMPLE, model, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
    return residuals
end

# Setup for all compressible algorithms
function setup_compressible_solvers(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
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
    mueffgradUt = VectorField(mesh)
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
     
    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset p_eqn.solver = _workspace(solvers.p.solver, _b(p_eqn))
  
    @info "Initialising energy model..."
    energyModel = initialise(model.energy, model, mdotf, rho, p_eqn, config)

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config;
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals    
end # end function

function CSIMPLE(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config ; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
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

    mugradUTx = FaceScalarField(mesh)
    mugradUTy = FaceScalarField(mesh)
    mugradUTz = FaceScalarField(mesh)

    divmugradUTx = ScalarField(mesh)
    divmugradUTy = ScalarField(mesh)
    divmugradUTz = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = KernelAbstractions.zeros(backend, TF, n_cells) 

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
    update_nueff!(nueff, nu, model.turbulence, config)
    @. mueff.values = nueff.values * rhof.values


    @info "Starting CSIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iteration ∈ 1:iterations
        time = iteration

        # gradU is updated in turbulence! function
        explicit_shear_stress!(
            mugradUTx, mugradUTy, mugradUTz, mueff, gradU, boundaries.U, config)
        div!(divmugradUTx, mugradUTx, config)
        div!(divmugradUTy, mugradUTy, config)
        div!(divmugradUTz, mugradUTz, config)
        
        @. mueffgradUt.x.values = divmugradUTx.values
        @. mueffgradUt.y.values = divmugradUTy.values
        @. mueffgradUt.z.values = divmugradUTz.values

        # Store previous values for next time step energy source terms
        @. model.energy.prevRhoK = rho.values*0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
        @. model.energy.prevP = p.values

        # Set up and solve momentum equations
        rx, ry, rz = solve_equation!(
            U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config
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
        if typeof(model.fluid) <: Compressible
            rp = solve_equation!(
                p_eqn, p, boundaries.p, solvers.p, config; 
                ref=nothing, irelax=solvers.p.relax) # perform implicit relaxation
        elseif typeof(model.fluid) <: WeaklyCompressible
            rp = solve_equation!(p_eqn, p, boundaries.p, solvers.p, config; ref=nothing)
        end

        if !isnothing(solvers.p.limit)
            pmin = solvers.p.limit[1]; pmax = solvers.p.limit[2]
            clamp!(p.values, pmin, pmax)
        end

        if typeof(model.fluid) <: WeaklyCompressible
            explicit_relaxation!(p, prev, solvers.p.relax, config)
        end
        grad!(∇p, pf, p, boundaries.p, time, config)
        limit_gradient!(schemes.p.limiter, ∇p, p, config)

        # non-orthogonal correction
        for i ∈ 1:ncorrectors
            discretise!(p_eqn, p, config)
            apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time, config)
            setReference!(p_eqn, pref, 1, config)
            nonorthogonal_face_correction(p_eqn, ∇p, rhorDf, config)
            update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            rp = solve_system!(p_eqn, solvers.p, p, nothing, config)
            if typeof(model.fluid) <: WeaklyCompressible
                explicit_relaxation!(p, prev, solvers.p.relax, config)
            end

            grad!(∇p, pf, p, boundaries.p, time, config)
            project_grad_tangent!(∇p, boundaries.U, config)
            limit_gradient!(schemes.p.limiter, ∇p, p, config)
        end

        # Correct mass flux and cell velocity

        if typeof(model.fluid) <: Compressible
            @. mdotf.values += pconv.values*(pf.values)
        end
        correct_mass_flux!(mdotf, p_eqn, config)

        correct_velocity!(U, Hv, ∇p, rD, config)
        # interpolate!(Uf, U, config) # Careful: reusing Uf for interpolation
        # correct_boundaries!(Uf, U, boundaries.U, time, config)
        
        # Perform turbulence calculations and update eddy viscosity
        turbulence!(turbulenceModel, model, S, prev, time, config) 
        update_nueff!(nueff, nu, model.turbulence, config)

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
        if typeof(mesh) <: Mesh3
            Uz_convergence = rz <= solvers.U.convergence
        end

        if (R_ux[iteration] <= solvers.U.convergence && 
            R_uy[iteration] <= solvers.U.convergence && 
            Uz_convergence &&
            R_p[iteration] <= solvers.p.convergence &&
            turbulenceModel.state.converged)

            progress.n = iteration
            finish!(progress)
            @info "Simulation converged in $iteration iterations!"
            if !signbit(write_interval)
                save_output(model, outputWriter, iteration, time, config)
            end
            break
        end

        ProgressMeter.next!(
            progress, showvalues = [
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

function explicit_shear_stress!(mugradUTx::FaceScalarField, mugradUTy::FaceScalarField, mugradUTz::FaceScalarField, mueff, gradU, U_BCs, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; faces, boundary_cellsID) = mugradUTx.mesh

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces
    kernel! = _explicit_shear_stress_internal!(_setup(backend, workgroup, ndrange)...)
    kernel!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces, n_bfaces)
    KernelAbstractions.synchronize(backend)

    ndrange=n_bfaces
    kernel! = _explicit_shear_stress_boundaries!(_setup(backend, workgroup, ndrange)...)
    kernel!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces)
    KernelAbstractions.synchronize(backend)

    for BC ∈ U_BCs
        zero_explicit_stress!(BC, mugradUTx, mugradUTy, mugradUTz, backend, workgroup)
    end
end

zero_explicit_stress!(BC, mugradUTx, mugradUTy, mugradUTz, backend, workgroup) = nothing

function zero_explicit_stress!(
    BC::Union{Slip,Symmetry}, mugradUTx, mugradUTy, mugradUTz, backend, workgroup)
    (; IDs_range) = BC
    ndrange = length(IDs_range)
    ndrange == 0 && return nothing
    kernel! = _zero_explicit_stress!(_setup(backend, workgroup, ndrange)...)
    kernel!(mugradUTx, mugradUTy, mugradUTz, IDs_range)
    KernelAbstractions.synchronize(backend)
end

@kernel function _zero_explicit_stress!(mugradUTx, mugradUTy, mugradUTz, IDs_range)
    i = @index(Global)
    fID = IDs_range[i]
    mugradUTx[fID] = 0
    mugradUTy[fID] = 0
    mugradUTz[fID] = 0
end

@kernel function _explicit_shear_stress_internal!(
    mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces, n_bfaces)
    i = @index(Global)

    fID = i + n_bfaces
    face = faces[fID]
    (; area, normal, ownerCells) = face 
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    
    # Linear interpolation of gradU at the face
    gradUf = 0.5 * (gradU[cID1] + gradU[cID2])
    
    # Explicit part of the stress projection: mu * ( (grad U)^T . n - 2/3 * (div U) * n )
    divU = sum(diag(gradUf))
    projection = transpose(gradUf) * normal - (2/3 * divU) * normal
    
    mueffi = mueff[fID]
    mugradUTx[fID] = mueffi * projection[1] * area
    mugradUTy[fID] = mueffi * projection[2] * area
    mugradUTz[fID] = mueffi * projection[3] * area
end

@kernel function _explicit_shear_stress_boundaries!(
    mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces)
    fID = @index(Global)

    face = faces[fID]
    (; area, normal, ownerCells) = face 
    cID1 = ownerCells[1]
    gradUi = gradU[cID1]
    
    # Explicit part of the stress projection at boundary: mu * ( (grad U)^T . n - 2/3 * (div U) * n )
    divUi = sum(diag(gradUi))
    projection = transpose(gradUi) * normal - (2/3 * divUi) * normal
    
    mueffi = mueff[fID]
    mugradUTx[fID] = mueffi * projection[1] * area
    mugradUTy[fID] = mueffi * projection[2] * area
    mugradUTz[fID] = mueffi * projection[3] * area
end