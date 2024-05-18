export piso!

piso!(model_in, config; resume=true, pref=nothing) = begin

    R_ux, R_uy, R_uz, R_p, model = setup_incompressible_solvers(
        PISO, model_in, config;
        resume=true, pref=nothing
        )
        
    return R_ux, R_uy, R_uz, R_p, model
end

function PISO(
    model, ∇p, ux_eqn, uy_eqn, uz_eqn, p_eqn, turbulence, config, volumes ; resume, pref)
    
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

    @info "Staring PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations

        # X velocity calculations
        # @. prev = U.x.values
        discretise!(ux_eqn, U.x.values, config)
        uy_eqn.equation.A.nzVal .= ux_eqn.equation.A.nzVal
        uz_eqn.equation.A.nzVal .= ux_eqn.equation.A.nzVal
        # uy_eqn.equation.b .= ux_eqn.equation.b
        # uz_eqn.equation.b .= ux_eqn.equation.b
        uy_eqn.equation.b .= -∇p.result.y.values.*volumes .+ U.y.values.*volumes/runtime.dt
        uz_eqn.equation.b .= -∇p.result.z.values.*volumes .+ U.z.values.*volumes/runtime.dt
        apply_boundary_conditions!(ux_eqn, U.x.BCs, config)
        implicit_relaxation!(ux_eqn, U.x.values, solvers.U.relax, mesh, config)
        update_preconditioner!(ux_eqn.preconditioner, mesh, config)
        run!(ux_eqn, solvers.U, U.x, config)
        residual!(R_ux, ux_eqn.equation, U.x, iteration, config)

        # Y velocity calculations
        # @. prev = U.y.values
        # discretise!(uy_eqn, prev, config)
        apply_boundary_conditions!(uy_eqn, U.y.BCs, config)
        implicit_relaxation!(uy_eqn, U.y.values, solvers.U.relax, mesh, config)
        update_preconditioner!(uy_eqn.preconditioner, mesh, config)
        run!(uy_eqn, solvers.U, U.y, config)
        residual!(R_uy, uy_eqn.equation, U.y, iteration, config)

        # Z velocity calculations (3D Mesh only)
        if typeof(mesh) <: Mesh3
            # @. prev = U.z.values
            # discretise!(uz_eqn, prev, config)
            apply_boundary_conditions!(uz_eqn, U.z.BCs, config)
            implicit_relaxation!(uz_eqn, U.z.values, solvers.U.relax, mesh, config)
            update_preconditioner!(uz_eqn.preconditioner, mesh, config)
            run!(uz_eqn, solvers.U, U.z, config)
            residual!(R_uz, uz_eqn.equation, U.z, iteration, config)
        end
          
        # Pressure correction
        inverse_diagonal!(rD, ux_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(ux_eqn, uy_eqn, uz_eqn, ∇p, config)

        for i ∈ 1:2
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
    end # corrector loop end
        
        # for i ∈ eachindex(divUTx)
        #     vol = mesh.cells[i].volume
        #     divUTx = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][1,1]+ gradUT[i][1,2] + gradUT[i][1,3])*vol
        #     divUTy = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][2,1]+ gradUT[i][2,2] + gradUT[i][2,3])*vol
        # end
        
        # convergence = 1e-7

        # if (R_ux[iteration] <= convergence && 
        #     R_uy[iteration] <= convergence && 
        #     R_p[iteration] <= convergence)

        #     print(
        #         """
        #         \n\n\n\n\n
        #         Simulation converged! $iteration iterations in
        #         """)
        #         if !signbit(write_interval)
        #             model2vtk(model, @sprintf "timestep_%.6d" iteration)
        #         end
        #     break
        # end

        # co = courant_number(U, mesh, runtime) # MUST IMPLEMENT!!!!!!

        ProgressMeter.next!(
            progress, showvalues = [
                (:time,iteration*runtime.dt),
                # (:Courant,co),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p, R_p[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0
            model2vtk(model, @sprintf "timestep_%.6d" iteration)
        end

    end # end for loop
    model_out = adapt(CPU(), model)
    return R_ux, R_uy, R_uz, R_p, model_out
end