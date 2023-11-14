export dsimple!

function dsimple!(model, config; resume=true, pref=nothing) 

    @info "Extracting configuration and input fields..."
    (; U, h, p, nu, mesh) = model
    (; solvers, schemes, runtime) = config

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    # nuf = ConstantScalar(nu) # Implement constant field!
    rhorDf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    alphaeff = FaceScalarField(mesh)
    initialise!(rhorDf, 1.0)
    divHv = ScalarField(mesh)

    divK = ScalarField(mesh)

    @info "Defining models..."

    ux_eqn = (
        Divergence{schemes.U.divergence}(mdotf, U.x) 
        - Laplacian{schemes.U.laplacian}(nueff, U.x) 
        == 
        -Source(∇p.result.x)
    ) → Equation(mesh)
    
    uy_eqn = (
        Divergence{schemes.U.divergence}(mdotf, U.y) 
        - Laplacian{schemes.U.laplacian}(nueff, U.y) 
        == 
        -Source(∇p.result.y)
    ) → Equation(mesh)

    h_eqn = (
        Divergence{schemes.h.divergence}(mdotf, h)
        - Laplacian{schemes.h.laplacian}(alphaeff, h) 
        == -Source(divK)
    ) → Equation(mesh)

    p_eqn = (
        Laplacian{schemes.p.laplacian}(rhorDf, p) == Source(divHv)
    ) → Equation(mesh)

    @info "Initialising preconditioners..."
    
    @reset ux_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
    @reset uy_eqn.preconditioner = ux_eqn.preconditioner
    @reset h_eqn.preconditioner = set_preconditioner(
                    solvers.h.preconditioner, h_eqn, h.BCs, runtime)
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, runtime)

    @info "Pre-allocating solvers..."
     
    @reset ux_eqn.solver = solvers.U.solver(_A(ux_eqn), _b(ux_eqn))
    @reset uy_eqn.solver = solvers.U.solver(_A(uy_eqn), _b(uy_eqn))
    @reset h_eqn.solver = solvers.h.solver(_A(h_eqn), _b(h_eqn))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

    if isturbulent(model)
        @info "Initialising turbulence model..."
        turbulence = initialise_RANS(mdotf, p_eqn, config, model)
        config = turbulence.config
    else
        turbulence = nothing
    end

    R_ux, R_uy, R_h, R_p  = dSIMPLE_loop(
    model, ∇p, ux_eqn, uy_eqn, h_eqn, p_eqn, turbulence, config ; resume=resume, pref=pref)

    return R_ux, R_uy, R_h, R_p     
end # end function

function dSIMPLE_loop(
    model, ∇p, ux_eqn, uy_eqn, h_eqn, p_eqn, turbulence, config ; resume, pref)
    
    # Extract model variables and configuration
    (;mesh, U, h, p, nu) = model
    # ux_model, uy_model = ux_eqn.model, uy_eqn.model
    p_model = p_eqn.model
    (; solvers, schemes, runtime) = config
    (; iterations, write_interval) = runtime
    
    mdotf = get_flux(ux_eqn, 1)
    nueff = get_flux(ux_eqn, 2)
    alphaeff = get_flux(h_eqn, 2)
    rhorDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)
    divK = get_source(h_eqn, 1)

    # K = get_phi(h_eqn, 2)
    
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
    hf = FaceScalarField(mesh)
    pf = FaceScalarField(mesh)
    rhof = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rhorD = ScalarField(mesh)
    rho = ScalarField(mesh)
    phiKf = FaceVectorField(mesh)

    # Pre-allocate auxiliary variables

    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)  

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_h = ones(TF, iterations)
    R_p = ones(TF, iterations)

    R = 287.
    Cp = 1005.

    # T = h/Cp
    # rho = p/(R*T)

    interpolate!(hf, h)
    correct_boundaries!(hf, h, h.BCs)
   
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, U.BCs)
  
    grad!(∇p, pf, p, p.BCs)

    rhof.values .= (pf.values.*Cp)./(R.*hf.values)
    rho.values .= (p.values.*Cp)./(R.*h.values)
    
    flux!(mdotf, Uf, rhof)

    update_nueff!(nueff, nu, turbulence)

    update_alphaeff!(alphaeff, nu, turbulence)

    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations

        # I think divK is the problem
        phiKf.x.values .= rhof.values .* Uf.x.values .* 0.5 .* (Uf.x.values .* Uf.x.values .+ Uf.y.values .* Uf.y.values .+ Uf.z.values .* Uf.z.values)
        phiKf.y.values .= rhof.values .* Uf.y.values .* 0.5 .* (Uf.x.values .* Uf.x.values .+ Uf.y.values .* Uf.y.values .+ Uf.z.values .* Uf.z.values)
        phiKf.z.values .= rhof.values .* Uf.z.values .* 0.5 .* (Uf.x.values .* Uf.x.values .+ Uf.y.values .* Uf.y.values .+ Uf.z.values .* Uf.z.values)
        explicitdiv!(divK, phiKf)
        divK.values .= 0

        @. prev = U.x.values
        discretise!(ux_eqn, prev, runtime)
        apply_boundary_conditions!(ux_eqn, U.x.BCs)
        # ux_eqn.b .-= divUTx
        implicit_relaxation!(ux_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(ux_eqn.preconditioner)
        run!(ux_eqn, solvers.U) #opP=Pu.P, solver=solver_U)
        residual!(R_ux, ux_eqn.equation, U.x, iteration)

        @. prev = U.y.values
        discretise!(uy_eqn, prev, runtime)
        apply_boundary_conditions!(uy_eqn, U.y.BCs)
        # uy_eqn.b .-= divUTy
        implicit_relaxation!(uy_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(uy_eqn.preconditioner)
        run!(uy_eqn, solvers.U)
        residual!(R_uy, uy_eqn.equation, U.y, iteration)
        
        @. prev = h.values
        discretise!(h_eqn, prev, runtime)
        apply_boundary_conditions!(h_eqn, h.BCs)
        implicit_relaxation!(h_eqn.equation, prev, solvers.h.relax)
        update_preconditioner!(h_eqn.preconditioner)
        run!(h_eqn, solvers.h)
        residual!(R_h, h_eqn.equation, h, iteration)

        # interpolate!(hf, h)
        # correct_boundaries!(hf, h, h.BCs)
        # interpolate!(pf, p)
        # correct_boundaries!(pf, p, p.BCs)
        # rhof.values .= (pf.values.*Cp)./(R.*hf.values)
        
        # rho.values .= (p.values.*Cp)./(R.*h.values)

        inverse_diagonal!(rhorD, ux_eqn.equation)
        rhorD.values .*= rho.values
        interpolate!(rhorDf, rhorD)
        remove_pressure_source!(ux_eqn, uy_eqn, ∇p)
        H!(Hv, U, ux_eqn, uy_eqn)

        Hv.x.values .*= rho.values
        Hv.y.values .*= rho.values
        Hv.z.values .*= rho.values

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

        # rhorD.values .*= rho.values
        correct_velocity!(U, Hv, ∇p, rhorD)

        # @. prev = rho.values

        rho.values .= (p.values.*Cp)./(R.*h.values)
        interpolate!(hf, h)
        correct_boundaries!(hf, h, h.BCs)
        rhof.values .= (pf.values.*Cp)./(R.*hf.values)

        U.x.values ./= rho.values
        U.y.values ./= rho.values
        U.z.values ./= rho.values

        interpolate!(Uf, U)   
        correct_boundaries!(Uf, U, U.BCs)

        flux!(mdotf, Uf, rhof)

      

        # explicit_relaxation!(rho, prev, 0.01)
        

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

        convergence = 1e-20

        if (R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence && 
            R_h[iteration] <= convergence &&
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
                (:h, R_h[iteration]),
                (:p, R_p[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0
            model2vtk(model, @sprintf "iteration_%.6d" iteration)
        end

    end # end for loop
    return R_ux, R_uy, R_h, R_p 
end