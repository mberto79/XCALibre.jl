export piso!

piso!(model_in, config; resume=true, pref=nothing) = begin

    R_ux, R_uy, R_uz, R_p, model = setup_incompressible_solvers(
        PISO, model_in, config;
        resume=true, pref=nothing
        )
        
    return R_ux, R_uy, R_uz, R_p, model
end

function PISO(
    model, ∇p, U_eqn, p_eqn, config; resume=resume, pref=pref)
    
    # Extract model variables and configuration
    # (;mesh, U, p, nu) = model
    (; U, p) = model.momentum
    mesh = model.domain
    nu = _nu(model.fluid)
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware
    
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
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

    update_nueff!(nueff, nu, model.turbulence, config)


    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Staring PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations

        solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)
          
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        
        for i ∈ 1:2
            H!(Hv, U, U_eqn, config)
            
            # Interpolate faces
            interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
            correct_boundaries!(Uf, Hv, U.BCs, config)
            div!(divHv, Uf, config)
            
            # Pressure calculations
            @. prev = p.values
            solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)

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
                    solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)
                    grad!(∇p, pf, p, pBCs) 
                end
            end

            # Velocity and boundaries correction
            correct_velocity!(U, Hv, ∇p, rD, config)
            interpolate!(Uf, U, config)
            correct_boundaries!(Uf, U, U.BCs, config)
            flux!(mdotf, Uf, config)

            grad!(gradU, Uf, U, U.BCs, config)
            turbulence!(model, S, S2, prev, config) 
            update_nueff!(nueff, nu, model.turbulence, config)
    end # corrector loop end

    residual!(R_ux, U_eqn, U.x, iteration, xdir, config)
    residual!(R_uy, U_eqn, U.y, iteration, ydir, config)
    if typeof(mesh) <: Mesh3
        residual!(R_uz, U_eqn, U.z, iteration, zdir, config)
    end
    residual!(R_p, p_eqn, p, iteration, nothing, config)
        
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