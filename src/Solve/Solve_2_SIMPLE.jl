export isimple!, flux!

function isimple!(
    mesh::Mesh2{TI,TF}, nu, U, p, 
    setup_U, setup_p, iterations
    ; resume=true, pref=nothing) where {TI,TF}

    @info "Preallocating fields..."
    
    ∇p = Grad{Linear}(p)
    mdotf = FaceScalarField(mesh)
    nuf = ConstantScalar(nu) # Implement constant field! Priority 1
    rDf = FaceScalarField(mesh)
    rDf.values .= 1.0
    divHv_new = ScalarField(mesh)

    @info "Defining models..."

    ux_model = (
        Divergence{Linear}(mdotf, U.x) - Laplacian{Linear}(nuf, U.x) 
        == 
        Source(∇p.result.x)
    )
    
    uy_model = (
        Divergence{Linear}(mdotf, U.y) - Laplacian{Linear}(nuf, U.y) 
        == 
        Source(∇p.result.y)
    )

    p_model = (
        Laplacian{Linear}(rDf, p) == Source(divHv_new)
    )

    @info "Allocating matrix equations..."

    ux_eqn  = Equation(mesh)
    uy_eqn  = Equation(mesh)
    p_eqn    = Equation(mesh)

    @info "Initialising preconditioners..."

    # Pu = set_preconditioner(NormDiagonal(), ux_eqn, ux_model, uxBCs)
    # Pu = set_preconditioner(Jacobi(), ux_eqn, ux_model, uxBCs)
    # Pu = set_preconditioner(ILU0(), ux_eqn, ux_model, uxBCs)
    Pu = set_preconditioner(DILU(), ux_eqn, ux_model, U.x.BCs)
    Pp = set_preconditioner(LDL(), p_eqn, p_model, p.BCs)

    @info "Initialising linear solvers..."

    solver_p = setup_p.solver(p_eqn.A, p_eqn.b)
    solver_U = setup_U.solver(ux_eqn.A, ux_eqn.b)

    R_ux, R_uy, R_p  = SIMPLE_loop(
    mesh::Mesh2{TI,TF}, U, p, ∇p,
    setup_U, setup_p, iterations,
    ux_model, uy_model, p_model,
    Pu, Pp,
    solver_U, solver_p,
    ux_eqn, uy_eqn, p_eqn
    ; resume=true, pref=nothing)

    return R_ux, R_uy, R_p     
end # end function

function SIMPLE_loop(
    mesh::Mesh2{TI,TF}, U, p, ∇p,
    setup_U, setup_p, iterations,
    ux_model, uy_model, p_model,
    Pu, Pp,
    solver_U, solver_p,
    ux_eqn, uy_eqn, p_eqn
    ; resume=true, pref=nothing) where {TI,TF}

    
    # Extract model variables
    
    mdotf = get_flux(ux_model, 1)
    rDf = get_flux(p_model, 1)
    # nuf = ux_model.terms[2].flux
    divHv_new = ScalarField(p_model.sources[1].field, mesh, p.BCs)
    
    @info "Allocating working memory..."

    # Define aux fields 

    n_cells = length(mesh.cells)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    # ∇p = Grad{Midpoint}(p)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    # Hvf = FaceVectorField(mesh)
    Hv_flux = FaceScalarField(mesh)
    rD = ScalarField(mesh)

    # Pre-allocate auxiliary variables

    ux0 = zeros(TF, n_cells)
    uy0 = zeros(TF, n_cells)
    p0 = zeros(TF, n_cells)  

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_p = ones(TF, iterations)

    #### IMPLEMENT A SENSIBLE INITIALISATION TO INCLUDE WARM START!!!!
    # Update initial (guessed) fields

    @inbounds ux0 .= U.x.values
    @inbounds uy0 .= U.y.values 
    @inbounds p0 .= p.values
    
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, U.BCs)
    flux!(mdotf, Uf)
    source!(∇p, pf, p, p.BCs)

    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations
    # for iteration ∈ 1:iterations
        
        source!(∇p, pf, p, p.BCs)
        neg!(∇p)

        discretise!(ux_eqn, ux_model)
        @turbo @. uy_eqn.A.nzval = ux_eqn.A.nzval # Avoid rediscretising
        apply_boundary_conditions!(ux_eqn, ux_model, U.x.BCs)
        implicit_relaxation!(ux_eqn, ux0, setup_U.relax)
        update_preconditioner!(Pu)

        run!(ux_eqn, ux_model, setup_U, opP=Pu.P, solver=solver_U)
        residual!(R_ux, ux_eqn, U.x, iteration)

        @turbo @. uy_eqn.b = 0.0
        # discretise!(uy_eqn, uy_model)
        apply_boundary_conditions!(uy_eqn, uy_model, U.y.BCs)
        implicit_relaxation!(uy_eqn, uy0, setup_U.relax)
        update_preconditioner!(Pu)

        run!(uy_eqn, uy_model, setup_U, opP=Pu.P, solver=solver_U)
        residual!(R_uy, uy_eqn, U.y, iteration)
        
        inverse_diagonal!(rD, ux_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(ux_eqn, uy_eqn, ∇p, rD)
        H!(Hv, U, ux_eqn, uy_eqn, rD)
        
        # interpolate!(Hvf, Hv)
        # correct_boundaries!(Hvf, Hv, U.BCs)
        # flux!(Hv_flux, Hvf)
        # div!(divHv_new, Hv_flux)

        interpolate!(Uf, Hv)
        correct_boundaries!(Uf, Hv, U.BCs)
        flux!(Hv_flux, Uf)
        div!(divHv_new, Hv_flux)
   
        discretise!(p_eqn, p_model)
        apply_boundary_conditions!(p_eqn, p_model, p.BCs)
        setReference!(p_eqn, pref, 1)
        update_preconditioner!(Pp)
        run!( p_eqn, p_model, setup_p, opP=Pp.P, solver=solver_p)

        explicit_relaxation!(p, p0, setup_p.relax)
        residual!(R_p, p_eqn, p, iteration)

        grad!(∇p, pf, p, p.BCs) 
        # source!(∇p, pf, p, pBCs)

        correct = false
        if correct
            ncorrectors = 1
            for i ∈ 1:ncorrectors
                discretise!(p_eqn, p_model)
                apply_boundary_conditions!(p_eqn, p_model, p.BCs)
                setReference!(p_eqn, pref, 1)
                # grad!(∇p, pf, p, pBCs) 
                interpolate!(gradpf, ∇p, p)
                nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                correct!(p_eqn, p_model.terms.term1, pf)
                run!(p_eqn, p_model, setup_p, opP=Pp.P, solver=solver_p)
                grad!(∇p, pf, p, pBCs) 
            end
        end

        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, U.BCs)
        flux!(mdotf, Uf)

        @inbounds for i ∈ eachindex(ux0)
            ux0[i] = U.x.values[i]
            uy0[i] = U.y.values[i]
        end

        convergence = 1e-7

        if (R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence && 
            R_p[iteration] <= convergence)

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
        # DV = D/volume
        values[i] = volume/D
        # values[i] = 1.0/DV
        # values[i] = 1.0/view(A, i, i)[1]
    end
end

function explicit_relaxation!(phi, phi0, alpha)
    # @. phi.values = alpha*phi.values + (1.0 - alpha)*phi0
    # @. phi0 = phi.values
    values = phi.values
    @inbounds @simd for i ∈ eachindex(values)
        values[i] = phi0[i] + alpha*(values[i] - phi0[i])
        phi0[i] = values[i]
    end
end

function correct_velocity!(U, Hv, ∇p, rD)
    # @. U.x = Hv.x - ∇p.result.x*rD.values
    # @. U.y = Hv.y - ∇p.result.y*rD.values
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.result.x; dpdy = ∇p.result.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(Ux)
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    # @. ux.values = Hv.x - ∇p.result.x*rD.values
    # @. uy.values = Hv.y - ∇p.result.y*rD.values
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

function remove_pressure_source!(ux_eqn, uy_eqn, ∇p, rD)
    # @. ux_eqn.b -= ∇p.result.x/rD.values
    # @. uy_eqn.b -= ∇p.result.y/rD.values
    dpdx, dpdy, rD = ∇p.result.x, ∇p.result.y, rD.values
    bx, by = ux_eqn.b, uy_eqn.b
    @inbounds for i ∈ eachindex(bx)
        # rDi = rD[i]
        # bx[i] -= dpdx[i]/rDi
        # by[i] -= dpdy[i]/rDi
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

H!(Hv, v::VF, xeqn, yeqn, rD) where VF<:VectorField = 
begin
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))
    
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
        rD_temp = 1.0/D
        # rD_temp = rD[cID]
        x[cID] = (bx[cID] - sumx)*rD_temp
        y[cID] = (by[cID] - sumy)*rD_temp
        z[cID] = zero(F)
    end
end