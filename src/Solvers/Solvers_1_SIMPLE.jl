export simple!

simple!(model_in, config; resume=true, pref=nothing) = begin
    R_ux, R_uy, R_uz, R_p, model = setup_incompressible_solvers(
        SIMPLE, model_in, config;
        resume=true, pref=nothing
        )

    return R_ux, R_uy, R_uz, R_p, model
end

# Setup for all incompressible algorithms
function setup_incompressible_solvers(
    solver_variant, 
    model_in, config; resume=true, pref=nothing
    ) 

    (; solvers, schemes, runtime, hardware) = config

    @info "Extracting configuration and input fields..."
    model = adapt(hardware.backend, model_in)
    # model = model_in
    (; U, p, nu, mesh) = model

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    divHv = ScalarField(mesh)

    @info "Defining models..."

    ux_eqn = (
        Time{schemes.U.time}(U.x)
        + Divergence{schemes.U.divergence}(mdotf, U.x) 
        - Laplacian{schemes.U.laplacian}(nueff, U.x) 
        == 
        -Source(∇p.result.x)
    ) → Equation(mesh)
    
    uy_eqn = (
        Time{schemes.U.time}(U.y)
        + Divergence{schemes.U.divergence}(mdotf, U.y) 
        - Laplacian{schemes.U.laplacian}(nueff, U.y) 
        == 
        -Source(∇p.result.y)
    ) → Equation(mesh)

    uz_eqn = (
        Time{schemes.U.time}(U.z)
        + Divergence{schemes.U.divergence}(mdotf, U.z) 
        - Laplacian{schemes.U.laplacian}(nueff, U.z) 
        == 
        -Source(∇p.result.z)
    ) → Equation(mesh)

    p_eqn = (
        Laplacian{schemes.p.laplacian}(rDf, p) == Source(divHv)
    ) → Equation(mesh)

    CUDA.allowscalar(false)

    @info "Initialising preconditioners..."

    @reset ux_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, ux_eqn, U.x.BCs, config)
    @reset uy_eqn.preconditioner = ux_eqn.preconditioner
    @reset uz_eqn.preconditioner = ux_eqn.preconditioner
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, config)

    if isturbulent(model)
        @info "Initialising turbulence model..."
        turbulence = initialise_RANS(mdotf, p_eqn, config, model)
        config = turbulence.config
    else
        turbulence = nothing
    end

    @info "Pre-allocating solvers..."
     
    @reset ux_eqn.solver = solvers.U.solver(_A(ux_eqn), _b(ux_eqn))
    @reset uy_eqn.solver = solvers.U.solver(_A(uy_eqn), _b(uy_eqn))
    @reset uz_eqn.solver = solvers.U.solver(_A(uz_eqn), _b(uz_eqn))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

    R_ux, R_uy, R_uz, R_p, model  = solver_variant(
    model, ∇p, ux_eqn, uy_eqn, uz_eqn, p_eqn, turbulence, config ; resume=resume, pref=pref)

    return R_ux, R_uy, R_uz, R_p, model    
end # end function

function SIMPLE(
    model, ∇p, ux_eqn, uy_eqn, uz_eqn, p_eqn, turbulence, config ; resume, pref)
    
    # Extract model variables and configuration
    (;mesh, U, p, nu) = model
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware
    
    mdotf = get_flux(ux_eqn, 2)
    nueff = get_flux(ux_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT)
    S2 = ScalarField(mesh)

    n_cells = length(mesh.cells)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend) 

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    
    # Initial calculations
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, U.BCs, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, p.BCs, config)

    update_nueff!(nueff, nu, turbulence, config)
    
    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations

        # X velocity calculations
        @. prev = U.x.values
        discretise!(ux_eqn, prev, config)
        apply_boundary_conditions!(ux_eqn, U.x.BCs, config)
        implicit_relaxation!(ux_eqn, prev, solvers.U.relax, mesh, config)
        update_preconditioner!(ux_eqn.preconditioner, mesh, config)
        run!(ux_eqn, solvers.U, U.x, config)
        residual!(R_ux, ux_eqn.equation, U.x, iteration, config)

        # Y velocity calculations
        @. prev = U.y.values
        discretise!(uy_eqn, prev, config)
        apply_boundary_conditions!(uy_eqn, U.y.BCs, config)
        implicit_relaxation!(uy_eqn, prev, solvers.U.relax, mesh, config)
        update_preconditioner!(uy_eqn.preconditioner, mesh, config)
        run!(uy_eqn, solvers.U, U.y, config)
        residual!(R_uy, uy_eqn.equation, U.y, iteration, config)

        # Z velocity calculations (3D Mesh only)
        if typeof(mesh) <: Mesh3
            @. prev = U.z.values
            discretise!(uz_eqn, prev, config)
            apply_boundary_conditions!(uz_eqn, U.z.BCs, config)
            implicit_relaxation!(uz_eqn, prev, solvers.U.relax, mesh, config)
            update_preconditioner!(uz_eqn.preconditioner, mesh, config)
            run!(uz_eqn, solvers.U, U.z, config)
            residual!(R_uz, uz_eqn.equation, U.z, iteration, config)
        end
          
        # Pressure correction
        inverse_diagonal!(rD, ux_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(ux_eqn, uy_eqn, uz_eqn, ∇p, config)
        H!(Hv, U, ux_eqn, uy_eqn, uz_eqn, config)
        
        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs, config)
        div!(divHv, Uf, config)
        
        # Pressure calculations
        @. prev = p.values
        discretise!(p_eqn, prev, config)
        apply_boundary_conditions!(p_eqn, p.BCs, config)
        setReference!(p_eqn, pref, 1, config)
        update_preconditioner!(p_eqn.preconditioner, mesh, config)
        run!(p_eqn, solvers.p, p, config)

        # Relaxation and residual
        explicit_relaxation!(p, prev, solvers.p.relax, config)
        residual!(R_p, p_eqn.equation, p, iteration, config)

        # Gradient
        grad!(∇p, pf, p, p.BCs, config) 

        correct = false
        if correct
            ncorrectors = 1
            for i ∈ 1:ncorrectors
                discretise!(p_eqn)
                apply_boundary_conditions!(p_eqn, p.BCs)
                setReference!(p_eqn.equation, pref, 1)
                interpolate!(gradpf, ∇p, p)
                nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                correct!(p_eqn.equation, p_model.terms.term1, pf)
                run!(p_model, solvers.p)
                grad!(∇p, pf, p, pBCs) 
            end
        end

        # Velocity and boundaries correction
        correct_velocity!(U, Hv, ∇p, rD, config)
        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, U.BCs, config)
        flux!(mdotf, Uf, config)

        if isturbulent(model)
            grad!(gradU, Uf, U, U.BCs, config)
            turbulence!(turbulence, model, S, S2, prev, config) 
            update_nueff!(nueff, nu, turbulence, config)
        end
        
        convergence = 1e-7

        if (R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence && 
            # R_uz[iteration] <= convergence &&
            R_p[iteration] <= convergence)

            print(
                """
                \n\n\n\n\n
                Simulation converged! $iteration iterations in
                """)
                if !signbit(write_interval)
                    model2vtk(model, @sprintf "iteration_%.6d" iteration)
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
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0      
            model2vtk(model, @sprintf "iteration_%.6d" iteration)
        end

    end # end for loop
    model_out = adapt(CPU(), model)
    return R_ux, R_uy, R_uz, R_p, model_out
end