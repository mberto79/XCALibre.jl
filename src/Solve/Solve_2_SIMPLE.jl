export isimple!

function isimple!(
    mesh::Mesh2{TI,TF}, nu, U, p, k, ω, νt, 
    setup_U, setup_p, setup_turb, iterations
    ; resume=true, pref=nothing) where {TI,TF}

    @info "Preallocating fields..."
    
    ∇p = Grad{Linear}(p)
    mdotf = FaceScalarField(mesh)
    nuf = ConstantScalar(nu) # Implement constant field! Priority 1
    rDf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    divHv = ScalarField(mesh)

    @info "Allocating matrix equations..."

    ux_eqn  = Equation(mesh)
    uy_eqn  = Equation(mesh)
    eqn    = Equation(mesh)

    @info "Defining models..."

    ux_model = ux_eqn → (
        Divergence{Upwind}(mdotf, U.x) - Laplacian{Linear}(nueff, U.x) 
        == 
        Source(∇p.result.x)
    )
    
    uy_model = uy_eqn → (
        Divergence{Upwind}(mdotf, U.y) - Laplacian{Linear}(nueff, U.y) 
        == 
        Source(∇p.result.y)
    )

    p_model = eqn → (
        Laplacian{Linear}(rDf, p) == Source(divHv)
    ) 


    @info "Initialising turbulence model..."

    turbulence_model = initialise_RANS(k, ω, mdotf, eqn)

    @info "Initialising preconditioners..."

    # Pu = set_preconditioner(NormDiagonal(), ux_eqn, ux_model, U.x.BCs)
    # Pu = set_preconditioner(Jacobi(), ux_eqn, ux_model, U.x.BCs)
    Pu = set_preconditioner(ILU0(), ux_model, U.x.BCs)
    # Pu = set_preconditioner(DILU(), ux_eqn, ux_model, U.x.BCs)
    Pp = set_preconditioner(LDL(), p_model, p.BCs)

    @info "Initialising linear solvers..."

    solver_p = setup_p.solver(_A(p_model), _b(p_model))
    solver_U = setup_U.solver(_A(ux_model), _b(ux_model))

    R_ux, R_uy, R_p  = SIMPLE_loop(
    mesh::Mesh2{TI,TF}, U, p, nuf, νt, ∇p,
    setup_U, setup_p, setup_turb, iterations,
    ux_model, uy_model, p_model,
    turbulence_model,
    Pu, Pp,
    solver_U, solver_p,
    ; resume=true, pref=nothing)

    return R_ux, R_uy, R_p     
end # end function

function SIMPLE_loop(
    mesh::Mesh2{TI,TF}, U, p, nuf, νt, ∇p,
    setup_U, setup_p, setup_turb, iterations,
    ux_model, uy_model, p_model,
    turbulence_model,
    Pu, Pp,
    solver_U, solver_p,
    ; resume=true, pref=nothing) where {TI,TF}
    
    # Extract model variables
    
    mdotf = get_flux(ux_model, 1)
    nueff = get_flux(ux_model, 2)
    rDf = get_flux(p_model, 1)
    divHv = ScalarField(p_model.sources[1].field, mesh, p.BCs)
    
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

    prev = zeros(TF, n_cells)  

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_p = ones(TF, iterations)
    
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, U.BCs)
    flux!(mdotf, Uf)
    source!(∇p, pf, p, p.BCs)

    update_nueff!(nueff, nuf, turbulence_model)

    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations
        
        source!(∇p, pf, p, p.BCs)
        neg!(∇p)

        discretise!(ux_model)
        # @turbo @. uy_eqn.A.nzval = ux_eqn.A.nzval # Avoid rediscretising
        @inbounds ux_model.equation.b .+= ux_model.sources[1].field # should be moved out to "add_sources" function using the "Model" struct
        apply_boundary_conditions!(ux_model, U.x.BCs)
        # ux_eqn.b .-= divUTx
        @. prev = U.x.values
        implicit_relaxation!(ux_model.equation, prev, setup_U.relax)
        update_preconditioner!(Pu)
        run!(ux_model, setup_U, opP=Pu.P, solver=solver_U)
        residual!(R_ux, ux_model.equation, U.x, iteration)

        # @turbo @. uy_eqn.b = 0.0
        discretise!(uy_model)
        @inbounds uy_model.equation.b .+= uy_model.sources[1].field
        apply_boundary_conditions!(uy_model, U.y.BCs)
        # uy_eqn.b .-= divUTy
        @. prev = U.y.values
        implicit_relaxation!(uy_model.equation, prev, setup_U.relax)
        update_preconditioner!(Pu)

        run!(uy_model, setup_U, opP=Pu.P, solver=solver_U)
        residual!(R_uy, uy_model.equation, U.y, iteration)
        
        inverse_diagonal!(rD, ux_model.equation)
        interpolate!(rDf, rD)
        remove_pressure_source!(ux_model, uy_model, ∇p)
        H!(Hv, U, ux_model, uy_model)

        interpolate!(Uf, Hv) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs)
        div!(divHv, Uf)
   
        discretise!(p_model)
        @inbounds p_model.equation.b .+= p_model.sources[1].field
        apply_boundary_conditions!(p_model, p.BCs)
        setReference!(p_model.equation, pref, 1)
        update_preconditioner!(Pp)
        @. prev = p.values
        run!(p_model, setup_p, opP=Pp.P, solver=solver_p)

        explicit_relaxation!(p, prev, setup_p.relax)
        residual!(R_p, p_model.equation, p, iteration)

        grad!(∇p, pf, p, p.BCs) 
        # source!(∇p, pf, p, pBCs)

        correct = false
        if correct
            ncorrectors = 1
            for i ∈ 1:ncorrectors
                discretise!(p_model)
                apply_boundary_conditions!(p_model, p.BCs)
                setReference!(p_model.equation, pref, 1)
                # grad!(∇p, pf, p, pBCs) 
                interpolate!(gradpf, ∇p, p)
                nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                correct!(p_model.equation, p_model.terms.term1, pf)
                run!(p_model, setup_p, opP=Pp.P, solver=solver_p)
                grad!(∇p, pf, p, pBCs) 
            end
        end

        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, U.BCs)
        flux!(mdotf, Uf)

        grad!(gradU, Uf, U, U.BCs)
        
        
        turbulence!(
            turbulence_model, νt, nuf, S, S2, solver_p, setup_turb, prev, implicit_relaxation!
            ) 
        update_nueff!(nueff, nuf, turbulence_model)

        # for i ∈ eachindex(divUTx)
        #     vol = mesh.cells[i].volume
        #     divUTx = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][1,1]+ gradUT[i][1,2] + gradUT[i][1,3])*vol
        #     divUTy = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][2,1]+ gradUT[i][2,2] + gradUT[i][2,3])*vol
        # end
        
        convergence = 1e-6

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
    for i ∈ eachindex(nueff)
        nueff[i] = nu[i] + turb_model.νtf[i]
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

function implicit_relaxation!(eqn::E, field, alpha) where E<:Equation
    (; A, b) = eqn
    @inbounds for i ∈ eachindex(b)
        A[i,i] /= alpha
        b[i] += (1.0 - alpha)*A[i,i]*field[i]
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

function explicit_relaxation!(phi, phi0, alpha)
    @inbounds @simd for i ∈ eachindex(phi)
        phi[i] = phi0[i] + alpha*(phi[i] - phi0[i])
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

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    u = ux.values; v = uy.values; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.result.x; dpdy = ∇p.result.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(u)
        rDvalues_i = rDvalues[i]
        u[i] = Hvx[i] - dpdx[i]*rDvalues_i
        v[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function neg!(∇p)
    dpdx = ∇p.result.x; dpdy = ∇p.result.y
    @inbounds for i ∈ eachindex(dpdx)
        dpdx[i] *= -1.0
        dpdy[i] *= -1.0
    end
end

remove_pressure_source!(ux_model::M1, uy_model::M2, ∇p) where {M1,M2} = begin
    dpdx, dpdy = ∇p.result.x, ∇p.result.y
    bx, by = ux_model.equation.b, uy_model.equation.b
    @inbounds for i ∈ eachindex(bx)
        bx[i] -= dpdx[i]
        by[i] -= dpdy[i]
    end
end

function setReference!(pEqn::E, pRef, cellID) where E<:Equation
    if pRef === nothing
        return nothing
    else
        pEqn.b[cellID] += pEqn.A[cellID,cellID]*pRef
        pEqn.A[cellID,cellID] += pEqn.A[cellID,cellID]
    end
end

H!(Hv, v::VF, ux_model, uy_model) where VF<:VectorField = 
begin
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = ux_model.equation.A; Ay = uy_model.equation.A
    bx = ux_model.equation.b; by = uy_model.equation.b
    vx, vy = v.x, v.y
    F = eltype(v.x.values)
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*vx[nID]
            sumy += Ay[cID,nID]*vy[nID]
        end

        D = view(Ax, cID, cID)[1] # add check to use max of Ax or Ay)
        rD = 1.0/D
        x[cID] = (bx[cID] - sumx)*rD
        y[cID] = (by[cID] - sumy)*rD
        z[cID] = zero(F)
    end
end