export simple!

function simple!(model, config; resume=true, pref=nothing) 

    @info "Extracting configuration and input fields..."
    (; U, p, nu, mesh) = model
    (; solvers, schemes, runtime) = config

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    # nuf = ConstantScalar(nu) # Implement constant field!
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

    p_eqn = (
        Laplacian{schemes.p.laplacian}(rDf, p) == Source(divHv)
    ) → Equation(mesh)

    @info "Initialising preconditioners..."
    
    @reset ux_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
    @reset uy_eqn.preconditioner = ux_eqn.preconditioner
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, runtime)

    @info "Pre-allocating solvers..."
     
    @reset ux_eqn.solver = solvers.U.solver(_A(ux_eqn), _b(ux_eqn))
    @reset uy_eqn.solver = solvers.U.solver(_A(uy_eqn), _b(uy_eqn))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

    if isturbulent(model)
        @info "Initialising turbulence model..."
        turbulence = initialise_RANS(mdotf, p_eqn, config, model)
        config = turbulence.config
    else
        turbulence = nothing
    end

    R_ux, R_uy, R_p  = SIMPLE_loop(
    model, ∇p, ux_eqn, uy_eqn, p_eqn, turbulence, config ; resume=resume, pref=pref)

    return R_ux, R_uy, R_p     
end # end function

function SIMPLE_loop(
    model, ∇p, ux_eqn, uy_eqn, p_eqn, turbulence, config ; resume, pref)
    
    # Extract model variables and configuration
    (;mesh, U, p, nu) = model
    # ux_model, uy_model = ux_eqn.model, uy_eqn.model
    p_model = p_eqn.model
    (; solvers, schemes, runtime) = config
    (; iterations, write_interval) = runtime
    
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

    # Temp sources to test GradUT explicit source
    # divUTx = zeros(Float64, length(mesh.cells))
    # divUTy = zeros(Float64, length(mesh.cells))

    n_cells = length(mesh.cells)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    rDfx = FaceScalarField(mesh)
    rDfy = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    rDx = ScalarField(mesh)
    rDy = ScalarField(mesh)

    # Pre-allocate auxiliary variables

    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)  

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_p = ones(TF, iterations)
    
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, U.BCs)
    flux!(mdotf, Uf)
    grad!(∇p, pf, p, p.BCs)

    update_nueff!(nueff, nu, turbulence)

    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    ## define mesh for Quick Wall Correction (remove later)
    (; faces, cells, boundaries) = mesh  

    @time for iteration ∈ 1:iterations

        ## Quick Wall Correction
        for bci ∈ 1:length(U.x.BCs)
            if U.x.BCs[bci] isa Wall{}
                (; facesID, cellsID) = boundaries[U.x.BCs[bci].ID]
                @inbounds for i ∈ eachindex(cellsID)
                    faceID = facesID[i]
                    cellID = cellsID[i]
                    face = faces[faceID]
                    cell = cells[cellID]
                    (; area, normal, delta) = face 

                    Uc = U.x.values[cellID]
                    Vc = U.y.values[cellID]
                    nueff_face = nueff.values[faceID]

                    ux_eqn.equation.A[cellID, cellID] += nueff_face*area*(1-normal[1]*normal[1])/delta
                    uy_eqn.equation.A[cellID, cellID] += nueff_face*area*(1-normal[2]*normal[2])/delta

                    ux_eqn.equation.b[cellID] += nueff_face*area*((0)*(1-normal[1]*normal[1]) + (Vc-0)*(normal[2]*normal[1]))/delta
                    uy_eqn.equation.b[cellID] += nueff_face*area*((Uc-0)*(normal[1]*normal[2]) + (0)*(1-normal[2]*normal[2]))/delta
                end
            end
        end
        ## Enf of Quick Wall COrrectyion

        @. prev = U.x.values
        discretise!(ux_eqn, prev, runtime)
        apply_boundary_conditions!(ux_eqn, U.x.BCs)
        ## Quick Wall Correction
        for bci ∈ 1:length(U.x.BCs)
            if U.x.BCs[bci] isa Wall{}
                (; facesID, cellsID) = boundaries[U.x.BCs[bci].ID]
                @inbounds for i ∈ eachindex(cellsID)
                    faceID = facesID[i]
                    cellID = cellsID[i]
                    face = faces[faceID]
                    cell = cells[cellID]
                    (; area, normal, delta) = face 

                    Uc = U.x.values[cellID]
                    Vc = U.y.values[cellID]
                    nueff_face = nueff.values[faceID]

                    ux_eqn.equation.A[cellID, cellID] += nueff_face*area*(1-normal[1]*normal[1])/delta
                    uy_eqn.equation.A[cellID, cellID] += nueff_face*area*(1-normal[2]*normal[2])/delta

                    ux_eqn.equation.b[cellID] += nueff_face*area*((0)*(1-normal[1]*normal[1]) + (Vc-0)*(normal[2]*normal[1]))/delta
                    uy_eqn.equation.b[cellID] += nueff_face*area*((Uc-0)*(normal[1]*normal[2]) + (0)*(1-normal[2]*normal[2]))/delta
                end
            end
        end
        ## Enf of Quick Wall COrrectyion
        # ux_eqn.b .-= divUTx
        implicit_relaxation!(ux_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(ux_eqn.preconditioner)
        run!(ux_eqn, solvers.U) #opP=Pu.P, solver=solver_U)
        residual!(R_ux, ux_eqn.equation, U.x, iteration)

        @. prev = U.y.values
        discretise!(uy_eqn, prev, runtime)
        apply_boundary_conditions!(uy_eqn, U.y.BCs)
        ## Quick Wall Correction
        for bci ∈ 1:length(U.x.BCs)
            if U.x.BCs[bci] isa Wall{}
                (; facesID, cellsID) = boundaries[U.x.BCs[bci].ID]
                @inbounds for i ∈ eachindex(cellsID)
                    faceID = facesID[i]
                    cellID = cellsID[i]
                    face = faces[faceID]
                    cell = cells[cellID]
                    (; area, normal, delta) = face 

                    Uc = U.x.values[cellID]
                    Vc = U.y.values[cellID]
                    nueff_face = nueff.values[faceID]

                    ux_eqn.equation.A[cellID, cellID] += nueff_face*area*(1-normal[1]*normal[1])/delta
                    uy_eqn.equation.A[cellID, cellID] += nueff_face*area*(1-normal[2]*normal[2])/delta

                    ux_eqn.equation.b[cellID] += nueff_face*area*((0)*(1-normal[1]*normal[1]) + (Vc-0)*(normal[2]*normal[1]))/delta
                    uy_eqn.equation.b[cellID] += nueff_face*area*((Uc-0)*(normal[1]*normal[2]) + (0)*(1-normal[2]*normal[2]))/delta
                end
            end
        end
        ## Enf of Quick Wall COrrectyion
        # uy_eqn.b .-= divUTy
        implicit_relaxation!(uy_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(uy_eqn.preconditioner)
        run!(uy_eqn, solvers.U)
        residual!(R_uy, uy_eqn.equation, U.y, iteration)
        
        inverse_diagonal!(rDx, ux_eqn.equation)
        interpolate!(rDfx, rDx)
        inverse_diagonal!(rDy, uy_eqn.equation)
        interpolate!(rDfy, rDy)

        # for fci ∈ 1:length(rDfx)
        #     (; area, normal, delta) = faces[fci] 

        #     rDf.values[fci] = ((rDfx.values[fci]*normal[1])^2 + (rDfy.values[fci]*normal[2])^2)/((rDfx.values[fci]*normal[1]) + (rDfy.values[fci]*normal[2]))
        # end

        inverse_diagonal!(rD, ux_eqn.equation)
        interpolate!(rDf, rD)

        remove_pressure_source!(ux_eqn, uy_eqn, ∇p)
        H!(Hv, U, ux_eqn, uy_eqn)

        interpolate!(Uf, Hv) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs)
        div!(divHv, Uf)
   
        @. prev = p.values
        discretise!(p_eqn, prev, runtime)
        apply_boundary_conditions!(p_eqn, p.BCs)
        setReference!(p_eqn.equation, pref, 1)
        update_preconditioner!(p_eqn.preconditioner)
        run!(p_eqn, solvers.p)

        explicit_relaxation!(p, prev, solvers.p.relax)
        residual!(R_p, p_eqn.equation, p, iteration)

        grad!(∇p, pf, p, p.BCs) 

        correct = false
        if correct
            ncorrectors = 1
            for i ∈ 1:ncorrectors
                discretise!(p_eqn)
                apply_boundary_conditions!(p_eqn, p.BCs)
                setReference!(p_eqn.equation, pref, 1)
                # grad!(∇p, pf, p, pBCs) 
                interpolate!(gradpf, ∇p, p)
                nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                correct!(p_eqn.equation, p_model.terms.term1, pf)
                run!(p_model, solvers.p)
                grad!(∇p, pf, p, pBCs) 
            end
        end

        # correct_velocity!(U, Hv, ∇p, rD)
        correct_velocity_vec!(U, Hv, ∇p, rDx, rDy)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, U.BCs)
        flux!(mdotf, Uf)

        
        if isturbulent(model)
            grad!(gradU, Uf, U, U.BCs)
            turbulence!(turbulence, model, S, S2, prev) 
            update_nueff!(nueff, nu, turbulence)
        end
        
        # for i ∈ eachindex(divUTx)
        #     vol = mesh.cells[i].volume
        #     divUTx = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][1,1]+ gradUT[i][1,2] + gradUT[i][1,3])*vol
        #     divUTy = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][2,1]+ gradUT[i][2,2] + gradUT[i][2,3])*vol
        # end
        
        convergence = 1e-7

        if (R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence && 
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
                (:p, R_p[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0
            model2vtk(model, @sprintf "iteration_%.6d" iteration)
        end

    end # end for loop
    return R_ux, R_uy, R_p 
end