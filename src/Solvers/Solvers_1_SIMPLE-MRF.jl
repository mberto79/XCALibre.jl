export simple_MRF!

"""
    simple!(model_in, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0)

Incompressible variant of the SIMPLE algorithm to solving coupled momentum and mass conservation equations.

# Input arguments

- `model` reference to a `Physics` model defined by the user.
- `config` Configuration structure defined by the user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only (default = `nothing`)
- `ncorrectors` number of non-orthogonality correction loops (default = `0`)
- `inner_loops` number to inner loops used in transient solver based on PISO algorithm (default = `0`)

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`) with the following entries:

- `Ux` Vector of x-velocity residuals for each iteration.
- `Uy` Vector of y-velocity residuals for each iteration.
- `Uz` Vector of y-velocity residuals for each iteration.
- `p` Vector of pressure residuals for each iteration.

"""
function simple_MRF!(
    model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

    residuals = setup_incompressible_solvers_MRF(
        SIMPLE_MRF, model, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
    )

    return residuals
end

function setup_incompressible_solvers_MRF(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) 

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, p, Uf, pf) = model.momentum
    mesh = model.domain

    @info "Pre-allocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)
    omegaU = VectorField(mesh)

    @info "Defining models..."

    U_eqn = (
        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(nueff, U) 
        == 
        - Source(∇p.result)
        - Source(omegaU)
    ) → VectorEquation(U, boundaries.U)

    p_eqn = (
        - Laplacian{schemes.p.laplacian}(rDf, p) == - Source(divHv)
    ) → ScalarEquation(p, boundaries.p)

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p.preconditioner, p_eqn)

    @info "Pre-allocating solvers..."

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset p_eqn.solver = _workspace(solvers.p.solver, _b(p_eqn))

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, ∇p, U_eqn, p_eqn, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals
end # end function


function SIMPLE_MRF(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )
    
    # Extract model variables and configuration
    (; U, p, Uf, pf) = model.momentum
    (; nu, refFrames) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; iterations, write_interval,dt) = runtime
    (; backend) = hardware

    dt_cpu = zeros(_get_float(mesh), 1)
    copyto!(dt_cpu, config.runtime.dt)
    
    postprocess = convert_time_to_iterations(postprocess,model,dt_cpu[1],iterations)
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    omegaU = get_source(U_eqn, 2)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)

    outputWriter = initialise_writer(output, model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT, U, Uf)

    n_cells = length(mesh.cells)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = KernelAbstractions.zeros(backend, TF, n_cells) 

    # Pre-allocate vectors to hold residuals 
    R_ux = zeros(TF, iterations)
    R_uy = zeros(TF, iterations)
    R_uz = zeros(TF, iterations)
    R_p = zeros(TF, iterations)
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, boundaries.p, time, config)
    limit_gradient!(schemes.p.limiter, ∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)

    @info "Starting SIMPLE_MRF loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()



    for iteration ∈ 1:iterations
        time = iteration

        # Updates the OmegaU source term (function is defined below)
        update_mrf_sources!(omegaU, U, refFrames, config)

        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config)
        
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        correct_interpolation_periodic(rDf, rD, boundaries.U, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        
        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, boundaries.U, time, config)


        flux_mrf!(mdotf, Uf, config, refFrames)
        div!(divHv, mdotf, config)
        
        # Pressure calculations
        @. prev = p.values
        rp = solve_equation!(p_eqn, p, boundaries.p, solvers.p, config; ref=pref)
        explicit_relaxation!(p, prev, solvers.p.relax, config)
        
        grad!(∇p, pf, p, boundaries.p, time, config) 
        limit_gradient!(schemes.p.limiter, ∇p, p, config)

        # non-orthogonal correction
        for i ∈ 1:ncorrectors
            # @. prev = p.values
            discretise!(p_eqn, p, config)       
            apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time, config)
            # setReference!(p_eqn, pref, 1, config)
            nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
            # update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            rp = solve_system!(p_eqn, solvers.p, p, nothing, config)
            explicit_relaxation!(p, prev, solvers.p.relax, config)
            grad!(∇p, pf, p, boundaries.p, time, config) 
            limit_gradient!(schemes.p.limiter, ∇p, p, config)
        end

        # correct mass flux and velocity
        correct_mass_flux!(mdotf, p_eqn, config)
        correct_velocity!(U, Hv, ∇p, rD, config)

        turbulence!(turbulenceModel, model, S, prev, time, config) 
        update_nueff!(nueff, nu, model.turbulence, config)

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
                if refFrames.polar == false
                    save_output(model, outputWriter, iteration, time, config)
                elseif refFrames.polar == true
                    save_output_polar(model, outputWriter, iteration, time, config, refFrames.frames.x0[1], refFrames.frames.rotaxis[1], mask=refFrames.global_mask)
                end
                save_postprocessing(postprocess,iteration,time,mesh,outputWriter,config.boundaries)
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
                turbulenceModel.state.residuals...
                ]
            )

        runtime_postprocessing!(postprocess,iteration,iterations,S,time,config)
        
        if iteration%write_interval + signbit(write_interval) == 0      
            if refFrames.polar == false
                    save_output(model, outputWriter, iteration, time, config)
                elseif refFrames.polar == true
                    save_output_polar(model, outputWriter, iteration, time, config, refFrames.frames.x0[1], refFrames.frames.rotaxis[1], mask=refFrames.global_mask)
                end
            save_postprocessing(postprocess,iteration,time,mesh,outputWriter,config.boundaries)
        end

    end # end for loop
    
    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end

# MRF functions

function update_mrf_sources!(omegaU, U, reference_frames, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = U.mesh
    cells = mesh.cells 

    ndrange = length(cells)
    kernel! = _update_mrf_sources!(_setup(backend, workgroup, ndrange)...)
    kernel!(omegaU, U, reference_frames)
end

@kernel function _update_mrf_sources!(omegaU, U, reference_frames)
    cID = @index(Global)

    (; frames, global_mask) = reference_frames
    (; omega, rotaxis) = frames

    if global_mask[cID] != 0
        frameID = Int(global_mask[cID])
        Omega = omega[frameID]*rotaxis[frameID]
        omegaU[cID] = Omega × U[cID]
    end
end

function flux_mrf!(phif::FS, psif::FV, config, reference_frames) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phif)
    kernel! = _flux_mrf!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, psif, reference_frames)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _flux_mrf!(phif, psif, reference_frames)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = phif
        (; faces) = mesh
        (; frames, global_mask) = reference_frames
        (; omega, rotaxis, x0) = frames
    end

    @inbounds begin
        (; area, normal, ownerCells) = faces[i]
        Sf = area * normal
        if global_mask[ownerCells[1]] != 0
            frameID = Int(global_mask[ownerCells[1]])
            Omega = omega[frameID]*rotaxis[frameID]
            r = faces[i].centre - x0[frameID]
            values[i] = (psif[i] ⋅ Sf) - ((Omega × r ⋅ Sf))
        elseif global_mask[ownerCells[2]] != 0
            frameID = Int(global_mask[ownerCells[2]])
            Omega = omega[frameID]*rotaxis[frameID]
            r = faces[i].centre - x0[frameID]
            values[i] = (psif[i] ⋅ Sf) - ((Omega × r ⋅ Sf))
        else
            values[i] = (psif[i] ⋅ Sf)
        end
    end
end