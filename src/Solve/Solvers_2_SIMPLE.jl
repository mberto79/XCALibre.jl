export isimple!, flux!

function isimple!(
    mesh::Mesh2{TI,TF}, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations
    ; resume=true, pref=nothing) where {TI,TF}


    # Pre-allocate fields
    @info "Preallocating fields..."
    # ux = ScalarField(mesh)
    # uy = ScalarField(mesh)
    ∇p = Grad{Linear}(p)
    mdotf = FaceScalarField(mesh)
    nuf = ConstantScalar(nu) # Implement constant field! Priority 1
    rDf = FaceScalarField(mesh)
    rDf.values .= 1.0
    divHv = ScalarField(mesh)


    # Define models 
    @info "Defining models..."
    ux_model = (
        Divergence{Linear}(mdotf, U.x) - Laplacian{Linear}(nuf, U.x) 
        == 
        Source(∇p.x)
    )
    
    uy_model = (
        Divergence{Linear}(mdotf, U.y) - Laplacian{Linear}(nuf, U.y) 
        == 
        Source(∇p.y)
    )

    p_model = (
        Laplacian{Linear}(rDf, p) == Source(divHv)
    )

    # Define equations
    @info "Allocating matrix equations..."
    ux_eqn  = Equation(mesh)
    uy_eqn  = Equation(mesh)
    p_eqn    = Equation(mesh)

    n_cells = m = n = length(mesh.cells)

    # Define preconditioners and linear operators
    opAx = LinearOperator(ux_eqn.A)
    opAy = LinearOperator(uy_eqn.A)
    opAp = LinearOperator(p_eqn.A)

    @info "Initialising preconditioners..."
    # Pu = set_preconditioner(NormDiagonal(), ux_eqn, ux_model, uxBCs)
    # Pu = set_preconditioner(Jacobi(), ux_eqn, ux_model, uxBCs)
    # Pu = set_preconditioner(ILU0(), ux_eqn, ux_model, uxBCs)
    Pu = set_preconditioner(DILU(), ux_eqn, ux_model, uxBCs)
    Pp = set_preconditioner(LDL(), p_eqn, p_model, pBCs)

    @info "Initialising linear solvers..."
    solver_p = setup_p.solver(p_eqn.A, p_eqn.b)
    solver_U = setup_U.solver(ux_eqn.A, ux_eqn.b)

    R_ux, R_uy, R_p  = SIMPLE_loop(
    mesh::Mesh2{TI,TF}, velocity, U, p, ∇p,
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations,
    ux_model, uy_model, p_model,
    opAx, opAy, opAp, Pu, Pp,
    solver_U, solver_p,
    ux_eqn, uy_eqn, p_eqn
    ; resume=true, pref=nothing)

    return R_ux, R_uy, R_p     
end # end function

function SIMPLE_loop(
    mesh::Mesh2{TI,TF}, velocity, U, p, ∇p,
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations,
    model_ux, model_uy, model_p,
    opAx, opAy, opAp, Pu, Pp,
    solver_U, solver_p,
    ux_eqn, uy_eqn, p_eqn
    ; resume=true, pref=nothing) where {TI,TF}

    @info "Allocating working memory..."

    # Extract model variables
    # ux = model_ux.terms[1].phi
    mdotf = model_ux.terms[1].flux
    # uy = model_uy.terms[1].phi
    nuf = model_ux.terms[2].flux
    rDf = model_p.terms[1].flux 
    rDf.values .= 1.0
    divHv_new = ScalarField(model_p.sources[1].field, mesh)

    # Define aux fields 
    n_cells = m = n = length(mesh.cells)

    Uf = FaceVectorField(mesh)
    
    pf = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    
    Hv = VectorField(mesh)
    Hvf = FaceVectorField(mesh)
    Hv_flux = FaceScalarField(mesh)
    divHv = Div(Hv, FaceVectorField(mesh), zeros(TF, n_cells), mesh)
    rD = ScalarField(mesh)

    # Pre-allocated auxiliary variables
    ux0 = zeros(TF, n_cells)
    uy0 = zeros(TF, n_cells)
    p0 = zeros(TF, n_cells)

    # ux0 .= velocity[1]
    # uy0 .= velocity[2]
    # p0 .= zero(TF)

    

    #### NEED TO IMPLEMENT A SENSIBLE INITIALISATION TO INCLUDE WARM START!!!!
    # Update initial (guessed) fields
    @turbo U.x.values .= velocity[1]
    @turbo U.y.values .= velocity[2]
    @turbo ux0 .= U.x.values
    @turbo uy0 .= U.y.values
    @turbo p0 .= p.values
    # @inbounds ux.values .= velocity[1]
    # @inbounds uy.values .= velocity[2]
    # @turbo U.x .= ux.values #velocity[1]
    # @turbo U.y .= uy.values# velocity[2]
    # end

    # volume  = volumes(mesh)
    # rvolume  = 1.0./volume
    
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, UBCs)
    flux!(mdotf, Uf)

    source!(∇p, pf, p, pBCs)

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_p = ones(TF, iterations)

    # Perform SIMPLE loops 
    @info "Staring SIMPLE loops..."
    progress = Progress(iterations; dt=1.0, showspeed=true)
    @time for iteration ∈ 1:iterations
        
        source!(∇p, pf, p, pBCs)
        neg!(∇p)

        discretise!(ux_eqn, model_ux)
        @turbo @. uy_eqn.A.nzval = ux_eqn.A.nzval
        apply_boundary_conditions!(ux_eqn, model_ux, uxBCs)
        implicit_relaxation!(ux_eqn, ux0, setup_U.relax)
        update_preconditioner!(Pu)
        run!(
            ux_eqn, model_ux, uxBCs, 
            setup_U, opA=opAx, opP=Pu.P, solver=solver_U
        )
        residual!(R_ux, ux_eqn, U.x, opAx, solver_U, iteration)


        @turbo @. uy_eqn.b = 0.0
        apply_boundary_conditions!(uy_eqn, model_uy, uyBCs)
        implicit_relaxation!(uy_eqn, uy0, setup_U.relax)
        update_preconditioner!(Pu)

        run!(
            uy_eqn, model_uy, uyBCs, 
            setup_U, opA=opAy, opP=Pu.P, solver=solver_U
        )
        residual!(R_uy, uy_eqn, U.y, opAy, solver_U, iteration)


        # for i ∈ eachindex(ux0)
        #     ux0[i] = U.x[i]
        #     uy0[i] = U.y[i]
        #     # U.x[i] = ux.values[i]
        #     # U.y[i] = uy.values[i]
        # end
        
        inverse_diagonal!(rD, ux_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(ux_eqn, uy_eqn, ∇p, rD)
        # H!(Hv, U, ux_eqn, uy_eqn)
        H!(Hv, U.x, U.y, ux_eqn, uy_eqn, rD)
        
        # @turbo for i ∈ eachindex(ux0)
        #     U.x[i] = ux0[i]
        #     U.y[i] = uy0[i]
        # end

        # div!(divHv, UBCs) # 7 allocations
        # @turbo @. divHv.values *= rvolume
        
        interpolate!(Hvf, Hv)
        correct_boundaries!(Hvf, Hv, UBCs)
        flux!(Hv_flux, Hvf)
        div!(divHv_new, Hv_flux)
   
        discretise!(p_eqn, model_p)
        apply_boundary_conditions!(p_eqn, model_p, pBCs)
        setReference!(p_eqn, pref, 1)
        update_preconditioner!(Pp)
        run!( # 30 allocs
            p_eqn, model_p, pBCs, 
            setup_p, opA=opAp, opP=Pp.P, solver=solver_p
        )

        grad!(∇p, pf, p, pBCs) 
        correct = false
        if correct
            ncorrectors = 1
            for i ∈ 1:ncorrectors
                discretise!(p_eqn, model_p)
                apply_boundary_conditions!(p_eqn, model_p, pBCs)
                setReference!(p_eqn, pref, 1)
                interpolate!(gradpf, ∇p, p)
                nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                correct!(p_eqn, model_p.terms.term1, pf)
                run!(
                    p_eqn, model_p, pBCs, 
                    setup_p, opA=opAp, opP=opPP, solver=solver_p
                )
                grad!(∇p, pf, p, pBCs) 
            end
        end

        # source!(∇p, pf, p, pBCs)
        grad!(∇p, pf, p, pBCs) 
        
        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, UBCs)
        flux!(mdotf, Uf)

        for i ∈ eachindex(ux0)
            ux0[i] = U.x[i]
            uy0[i] = U.y[i]
            # U.x[i] = ux.values[i]
            # U.y[i] = uy.values[i]
        end

        
        explicit_relaxation!(p, p0, setup_p.relax)
        residual!(R_p, p_eqn, p, opAp, solver_p, iteration)

        grad!(∇p, pf, p, pBCs) 
        # correct_velocity!(U.x, U.y, Hv, ∇p, rD)

        

        convergence = 1e-7
        if R_ux[iteration] <= convergence && R_uy[iteration] <= convergence && R_p[iteration] <= convergence
            print(
                """
                \n\n\n\n\n
                Simulation converged! $iteration iterations in""")
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

function residual!(Residual, equation, phi, opA, solver, iteration)
    (; A, b, R, Fx) = equation
    values = phi.values
    # Option 1
    
    mul!(Fx, opA, values)
    @inbounds @. R = abs(Fx - b)
    # res = max(norm(R), eps())/abs(mean(values))
    res = sqrt(mean(R.^2))/abs(mean(values))

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

function calculate_residuals(
    divU::ScalarField, UxEqn::Equation, UyEqn::Equation, Ux::ScalarField, Uy::ScalarField)
    
    continuityError = abs(mean(-divU.values))

    solMean = mean(Ux.values)
    N = sum(
        abs.(UxEqn.A*(Ux.values .- solMean)) + 
        abs.(UxEqn.b .- UxEqn.A*ones(length(UxEqn.b))*solMean)
        )
    UxResidual = (1/N)*sum(abs.(UxEqn.b - UxEqn.A*Ux.values))

    solMean = mean(Uy.values)
    N = sum(
        abs.(UyEqn.A*(Uy.values .- solMean)) + 
        abs.(UyEqn.b .- UyEqn.A*ones(length(UyEqn.b))*solMean)
        )
    UyResidual = (1/N)*sum(abs.(UyEqn.b - UyEqn.A*Uy.values))

    return continuityError, UxResidual, UyResidual
end


function flux!(phif::FaceScalarField{TI,TF}, psif::FaceVectorField{TI,TF}) where {TI,TF}
    (; mesh, values) = phif
    (; faces) = mesh 
    @inbounds for fID ∈ eachindex(faces)
        (; area, normal) = faces[fID]
        Sf = area*normal
        values[fID] = psif[fID]⋅Sf
    end
end

function implicit_relaxation!(eqn::Equation{I,F}, field, alpha) where {I,F}
    (; A, b) = eqn
    @inbounds @simd for i ∈ eachindex(b)
        A[i,i] /= alpha
        b[i] += (1.0 - alpha)*A[i,i]*field[i]
    end
end

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

function inverse_diagonal!(rD::ScalarField{I,F}, eqn) where {I,F}
    (; mesh, values) = rD
    cells = mesh.cells
    A = eqn.A
    @inbounds for i ∈ eachindex(values)
        D = view(A, i, i)[1]
        volume = cells[i].volume
        values[i] = volume/D
    end
end

function explicit_relaxation!(phi, phi0, alpha)
    values = phi.values
    @inbounds @simd for i ∈ eachindex(values)
        values[i] = phi0[i] + alpha*(values[i] - phi0[i])
        phi0[i] = values[i]
    end
end

function correct_velocity!(U, Hv, ∇p, rD)
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(Ux)
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    u = ux.values; v = uy.values; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(u)
        rDvalues_i = rDvalues[i]
        u[i] = Hvx[i] - dpdx[i]*rDvalues_i
        v[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function neg!(∇p)
    dpdx = ∇p.x; dpdy = ∇p.y
    @inbounds for i ∈ eachindex(dpdx)
        dpdx[i] *= -1.0
        dpdy[i] *= -1.0
    end
end

function remove_pressure_source!(ux_eqn, uy_eqn, ∇p, rD)
    dpdx, dpdy, rD = ∇p.x, ∇p.y, rD.values
    bx, by = ux_eqn.b, uy_eqn.b
    @inbounds for i ∈ eachindex(bx)
        bx[i] -= dpdx[i]
        by[i] -= dpdy[i]
    end
end

function setReference!(pEqn::Equation{TI,TF}, pRef, cellID::TI) where {TI,TF}
    if pRef === nothing
        return nothing
    else
        pEqn.b[cellID] += pEqn.A[cellID,cellID]*pRef
        pEqn.A[cellID,cellID] += pEqn.A[cellID,cellID]
    end
end

function H!(Hv::VectorField, v::VectorField{I,F}, xeqn, yeqn, rD) where {I,F}
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))
    
    vx, vy = v.x, v.y
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*vx[nID]
            sumy += Ay[cID,nID]*vy[nID]
        end
        D = view(Ax, cID, cID)[1] # Good for now (add check to use max of Ax or Ay)
        rD_temp = 1.0/D
        x[cID] = (bx[cID] - sumx)*rD_temp
        y[cID] = (by[cID] - sumy)*rD_temp
        z[cID] = zero(F)
    end
end

function H!(
    Hv::VectorField, ux::ScalarField{I,F}, uy::ScalarField{I,F}, xeqn, yeqn, rD
    ) where {I,F}
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))
    ux_vals = ux.values
    uy_vals = uy.values
    
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*ux_vals[nID]
            sumy += Ay[cID,nID]*uy_vals[nID]
        end

        D = view(Ax, cID, cID)[1] # Good for now (add check to use max of Ax or Ay)
        rD_temp = 1.0/D
        x[cID] = (bx[cID] - sumx)*rD_temp
        y[cID] = (by[cID] - sumy)*rD_temp
        z[cID] = zero(F)
    end
end