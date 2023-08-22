export isimple!

function isimple!(
    model, config, iterations
    ; resume=true, pref=nothing) 

    @info "Extracting configuration and input fields..."
    (; U, p, nu, mesh) = model
    (; solvers, schemes) = config

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.U.gradient}(p)
    mdotf = FaceScalarField(mesh)
    # nuf = ConstantScalar(nu) # Implement constant field!
    rDf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    divHv = ScalarField(mesh)

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

    p_eqn = (
        Laplacian{schemes.p.laplacian}(rDf, p) == Source(divHv)
    ) → Equation(mesh)

    # ux_eqn = ModelEquation(ux_model, Equation(mesh), (), ())
    # uy_eqn = ModelEquation(uy_model, Equation(mesh), (), ())
    # p_eqn = ModelEquation(p_model, Equation(mesh),(), ())

    @info "Initialising preconditioners..."
    
    @reset ux_eqn.preconditioner = set_preconditioner(
                        solvers.U.preconditioner, ux_eqn, U.x.BCs)
    @reset uy_eqn.preconditioner = ux_eqn.preconditioner
    @reset p_eqn.preconditioner = set_preconditioner(
                        solvers.p.preconditioner, p_eqn, p.BCs)

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
    model, ∇p, iterations,
    ux_eqn, uy_eqn, p_eqn,
    turbulence, config
    ; resume=true, pref=nothing)

    return R_ux, R_uy, R_p     
end # end function

function SIMPLE_loop(
    model, ∇p, iterations,
    ux_eqn, uy_eqn, p_eqn,
    turbulence, config
    ; resume=true, pref=nothing)
    
    # Extract model variables and configuration
    (;mesh, U, p, nu) = model
    ux_model, uy_model = ux_eqn.model, uy_eqn.model
    p_model = p_eqn.model
    (; solvers, schemes) = config
    
    mdotf = get_flux(ux_eqn, 1)
    nueff = get_flux(ux_eqn, 2)
    rDf = get_flux(p_eqn, 1)
    # divHv = ScalarField(p_model.sources[1].field, mesh, p.BCs)
    divHv = get_source(p_eqn, 1)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{Linear}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT)
    S2 = ScalarField(mesh)

    # Temp sources to test GradUT explicit source
    # divUTx = zeros(Float64, length(mesh.cells))
    # divUTy = zeros(Float64, length(mesh.cells))

    n_cells = length(mesh.cells)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)

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

    @time for iteration ∈ 1:iterations
        

        discretise!(ux_eqn)
        apply_boundary_conditions!(ux_eqn, U.x.BCs)
        # ux_eqn.b .-= divUTx
        @. prev = U.x.values
        implicit_relaxation!(ux_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(ux_eqn.preconditioner)
        run!(ux_eqn, solvers.U) #opP=Pu.P, solver=solver_U)
        residual!(R_ux, ux_eqn.equation, U.x, iteration)

        # @turbo @. uy_eqn.b = 0.0
        discretise!(uy_eqn)
        # @inbounds uy_model.equation.b .+= uy_model.sources[1].field
        apply_boundary_conditions!(uy_eqn, U.y.BCs)
        # uy_eqn.b .-= divUTy
        @. prev = U.y.values
        implicit_relaxation!(uy_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(uy_eqn.preconditioner)
        run!(uy_eqn, solvers.U)
        residual!(R_uy, uy_eqn.equation, U.y, iteration)
        
        inverse_diagonal!(rD, ux_eqn.equation)
        interpolate!(rDf, rD)
        remove_pressure_source!(ux_eqn, uy_eqn, ∇p)
        H!(Hv, U, ux_eqn, uy_eqn)

        interpolate!(Uf, Hv) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs)
        div!(divHv, Uf)
   
        discretise!(p_eqn)
        apply_boundary_conditions!(p_eqn, p.BCs)
        setReference!(p_eqn.equation, pref, 1)
        update_preconditioner!(p_eqn.preconditioner)
        @. prev = p.values
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

        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, U.BCs)
        flux!(mdotf, Uf)

        grad!(gradU, Uf, U, U.BCs)

        if isturbulent(model)
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

    end # end for loop
    return R_ux, R_uy, R_p 
end

update_nueff!(nueff, nu, turb_model) = begin
    if turb_model === nothing
        for i ∈ eachindex(nueff)
            nueff[i] = nu[i]
        end
    else
        for i ∈ eachindex(nueff)
            nueff[i] = nu[i] + turb_model.νtf[i]
        end
    end
end

function residual!(Residual, equation, phi, iteration)
    (; A, b, R, Fx) = equation
    values = phi.values
    # Option 1
    
    mul!(Fx, A, values)
    @inbounds @. R = abs(Fx - b)
    # res = sqrt(mean(R.^2))/abs(mean(values))
    res = sqrt(mean(R.^2))/norm(b)


    # res = max(norm(R), eps())/abs(mean(values))

    # sum_mean = zero(TF)
    # sum_norm = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum_mean += values[i]
    #     sum_norm += abs(Fx[i] - b[i])^2
    # end
    # N = length(values)
    # res = sqrt(sum_norm)/abs(sum_mean/N)

    # Option 2
    # mul!(Fx, opA, values)
    # @inbounds @. R = b - Fx
    # sum = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum += (R[i])^2 
    # end
    # res = sqrt(sum/length(R))

    # Option 3 (OpenFOAM definition)
    # solMean = mean(values)
    # mul!(R, opA, values .- solMean)
    # term1 = abs.(R)
    # mul!(R, opA, values)
    # Fx .= b .- R.*solMean./values
    # term2 = abs.(Fx)
    # N = sum(term1 .+ term2)
    # res = (1/N)*sum(abs.(b .- R))

    # term1 = abs.(opA*(values .- solMean))
    # term2 = abs.(b .- R.*solMean./values)
    # N = sum(term1 .+ term2)
    # res = (1/N)*sum(abs.(b - opA*values))

    # print("Residual: ", res, " (", niterations(solver), " iterations)\n") 
    # @printf "\tResidual: %.4e (%i iterations)\n" res niterations(solver)
    # return res
    Residual[iteration] = res
    nothing
end

function flux!(phif::FS, psif::FV) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; mesh, values) = phif
    (; faces) = mesh 
    @inbounds for fID ∈ eachindex(faces)
        (; area, normal) = faces[fID]
        Sf = area*normal
        values[fID] = psif[fID]⋅Sf
    end
end

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

function inverse_diagonal!(rD::S, eqn) where S<:ScalarField
    (; mesh, values) = rD
    cells = mesh.cells
    A = eqn.A
    @inbounds for i ∈ eachindex(values)
        # D = view(A, i, i)[1]
        D = A[i,i]
        volume = cells[i].volume
        values[i] = volume/D
    end
end

function correct_velocity!(U, Hv, ∇p, rD)
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.result.x; dpdy = ∇p.result.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(Ux)
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

remove_pressure_source!(ux_eqn::M1, uy_eqn::M2, ∇p) where {M1,M2} = begin
    cells = get_phi(ux_eqn).mesh.cells
    source_sign = get_source_sign(ux_eqn, 1)
    dpdx, dpdy = ∇p.result.x, ∇p.result.y
    bx, by = ux_eqn.equation.b, uy_eqn.equation.b
    @inbounds for i ∈ eachindex(bx)
        volume = cells[i].volume
        bx[i] -= source_sign*dpdx[i]*volume
        by[i] -= source_sign*dpdy[i]*volume
    end
end

H!(Hv, v::VF, ux_eqn, uy_eqn) where VF<:VectorField = 
begin
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = ux_eqn.equation.A; Ay = uy_eqn.equation.A
    bx = ux_eqn.equation.b; by = uy_eqn.equation.b
    vx, vy = v.x, v.y
    F = eltype(v.x.values)
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours, volume) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*vx[nID]
            sumy += Ay[cID,nID]*vy[nID]
        end

        D = view(Ax, cID, cID)[1] # add check to use max of Ax or Ay)
        rD = 1.0/D
        # rD = volume/D
        x[cID] = (bx[cID] - sumx)*rD
        y[cID] = (by[cID] - sumy)*rD
        z[cID] = zero(F)
    end
end