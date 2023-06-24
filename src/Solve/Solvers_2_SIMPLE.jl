export isimple!, flux!

function isimple!(
    mesh::Mesh2{TI,TF}, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations
    ; resume=true, pref=nothing) where {TI,TF}

    n_cells = m = n = length(mesh.cells)

    # Pre-allocate fields
    ux = ScalarField(mesh)
    uy = ScalarField(mesh)
    # U = VectorField(mesh)
    Uf = FaceVectorField(mesh)
    # mdot = ScalarField(mesh)
    mdotf = FaceScalarField(mesh)
    nuf = ConstantScalar(nu) # Implement constant field! Priority 1
    pf = FaceScalarField(mesh)
    ∇p = Grad{Linear}(p)
    # ∇p = Grad{Midpoint}(p)
    gradpf = FaceVectorField(mesh)
    
    Hv = VectorField(mesh)
    Hvf = FaceVectorField(mesh)
    Hv_flux = FaceScalarField(mesh)
    divHv_new = ScalarField(mesh)
    divHv = Div(Hv, FaceVectorField(mesh), zeros(TF, n_cells), mesh)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)

    # Pre-allocated auxiliary variables
    ux0 = zeros(TF, n_cells)
    uy0 = zeros(TF, n_cells)
    p0 = zeros(TF, n_cells)

    ux0 .= velocity[1]
    uy0 .= velocity[2]
    p0 .= zero(TF)
    rDf.values .= 1.0

    # Define models 
    model_ux = (
        Divergence{Linear}(mdotf, ux) - Laplacian{Linear}(nuf, ux) 
        == 
        Source(∇p.x)
    )
    
    model_uy = (
        Divergence{Linear}(mdotf, uy) - Laplacian{Linear}(nuf, uy) 
        == 
        Source(∇p.y)
    )

    model_p = (
        Laplacian{Linear}(rDf, p) == Source(divHv_new)
    )

    # Define equations
    ux_eqn  = Equation(mesh)
    uy_eqn  = Equation(mesh)
    p_eqn    = Equation(mesh)

    # Define preconditioners and linear operators
    opAx = LinearOperator(ux_eqn.A)
    Px = ilu0(ux_eqn.A)
    opPUx = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Px, v))
    
    opAy = LinearOperator(uy_eqn.A)
    Py = ilu0(uy_eqn.A)
    opPUy = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Py, v))
    
    discretise!(p_eqn, model_p)
    apply_boundary_conditions!(p_eqn, model_p, pBCs)
    opAp = LinearOperator(p_eqn.A)
    opPP = opLDL(p_eqn.A)

    solver_p = setup_p.solver(p_eqn.A, p_eqn.b)
    solver_U = setup_U.solver(ux_eqn.A, ux_eqn.b)

    #### NEED TO IMPLEMENT A SENSIBLE INITIALISATION TO INCLUDE WARM START!!!!
    # Update initial (guessed) fields

    @turbo ux0 .= ux.values
    @turbo uy0 .= uy.values 
    @turbo p0 .= p.values
    @inbounds ux.values .= velocity[1]
    @inbounds uy.values .= velocity[2]
    @turbo U.x .= ux.values #velocity[1]
    @turbo U.y .= uy.values# velocity[2]
    # end
    volume  = volumes(mesh)
    rvolume  = 1.0./volume
    
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, UBCs)
    flux!(mdotf, Uf)

    source!(∇p, pf, p, pBCs)
    
    # R_ux = TF[]
    # R_uy = TF[]
    # R_p = TF[]

    R_ux = zeros(TF, iterations)
    R_uy = zeros(TF, iterations)
    R_p = zeros(TF, iterations)

    # Perform SIMPLE loops 
    @time for iteration ∈ 1:iterations

        print("\nIteration ", iteration, "\n") # 91 allocations
        
        # print("Solving Ux...")
        
        source!(∇p, pf, p, pBCs)
        neg!(∇p)

        discretise!(ux_eqn, model_ux)
        @turbo @. uy_eqn.A.nzval = ux_eqn.A.nzval
        apply_boundary_conditions!(ux_eqn, model_ux, uxBCs)
        implicit_relaxation!(ux_eqn, ux0, setup_U.relax)
        ilu0!(Px, ux_eqn.A)
        run!(
            ux_eqn, model_ux, uxBCs, 
            setup_U, opA=opAx, opP=opPUx, solver=solver_U
        )
        r_ux = residual(ux_eqn, ux, opAx, solver_U)


        # print("Solving Uy...")

        @turbo @. uy_eqn.b = 0.0
        apply_boundary_conditions!(uy_eqn, model_uy, uyBCs)
        implicit_relaxation!(uy_eqn, uy0, setup_U.relax)
        ilu0!(Py, uy_eqn.A)
        run!(
            uy_eqn, model_uy, uyBCs, 
            setup_U, opA=opAy, opP=opPUy, solver=solver_U
        )
        r_uy = residual(uy_eqn, uy, opAy, solver_U)


        @turbo for i ∈ eachindex(ux0)
            ux0[i] = U.x[i]
            uy0[i] = U.y[i]
            # U.x[i] = ux.values[i]
            # U.y[i] = uy.values[i]
        end
        
        inverse_diagonal!(rD, ux_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(ux_eqn, uy_eqn, ∇p, rD)
        # H!(Hv, U, ux_eqn, uy_eqn)
        H!(Hv, ux, uy, ux_eqn, uy_eqn, rD)
        
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
        # @turbo @. divHv_new.values *= rvolume

        # @inbounds @. rD.values *= volume
        # interpolate!(rDf, rD)
        # @inbounds @. rD.values *= rvolume

        # print("Solving p...")

        
        discretise!(p_eqn, model_p)
        apply_boundary_conditions!(p_eqn, model_p, pBCs)
        setReference!(p_eqn, pref, 1)
        run!(
            p_eqn, model_p, pBCs, 
            setup_p, opA=opAp, opP=opPP, solver=solver_p
        )

        grad!(∇p, pf, p, pBCs) 
        correct = false
        if correct
            ncorrectors = 1
            for i ∈ 1:ncorrectors
                discretise!(p_eqn, model_p)
                apply_boundary_conditions!(p_eqn, model_p, pBCs)
                setReference!(p_eqn, pref, 1)
                # grad!(∇p, pf, p, pBCs) 
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
        
        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, UBCs)
        flux!(mdotf, Uf)

        
        explicit_relaxation!(p, p0, setup_p.relax)
        r_p = residual(p_eqn, p, opAp, solver_p)

        # source!(∇p, pf, p, pBCs)
        grad!(∇p, pf, p, pBCs) 
        correct_velocity!(ux, uy, Hv, ∇p, rD)

        # push!(R_ux, r_ux)
        # push!(R_uy, r_uy)
        # push!(R_p, r_p)
        R_ux[iteration] = r_ux
        R_uy[iteration] = r_uy
        R_p[iteration] = r_p
        convergence = 1e-7
        if r_ux <= convergence && r_uy <= convergence && r_p <= convergence
            print("\nSimulation converged!\n")
            break
        end
    end # end for loop
    return R_ux, R_uy, R_p     
end # end function


function residual(equation::Equation{TI,TF}, phi, opA, solver) where {TI,TF}
    (; A, b, R, Fx) = equation
    values = phi.values
    # Option 1
    # mul!(Fx, opA, values)
    # @inbounds @. R = abs(Fx - b)
    # res = norm(R)/mean(values)

    # Option 2
    # mul!(Fx, opA, values)
    # @inbounds @. R = b - Fx
    # sum = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum += (R[i])^2 
    # end
    # res = sqrt(sum/length(R))

    # Option 3 (OpenFOAM definition)
    solMean = mean(values)
    term1 = abs.(opA*(values .- solMean))
    term2 = abs.(b .- opA*solMean*values./values)
    N = sum(term1 + term2)
    res = (1/N)*sum(abs.(b - opA*values))

    # print("Residual: ", res, " (", niterations(solver), " iterations)\n") 
    # @printf "\tResidual: %.4e (%i iterations)\n" res niterations(solver)
    return res
end

function calculate_residuals(
    divU::ScalarField, UxEqn::Equation, UyEqn::Equation, Ux::ScalarField, Uy::ScalarField)
    
    continuityError = abs(mean(-divU.values))
    # UxResidual = norm(UxEqn.A*Ux.values - UxEqn.b)/Rx₀
    # UyResidual = norm(UyEqn.A*Uy.values - UyEqn.b)/Ry₀

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

    # UxResidual = sum(sqrt.((UxEqn.b - UxEqn.A*Ux.values).^2))/length(UxEqn.b)/Rx₀
    # UyResidual = sum(sqrt.((UyEqn.b - UyEqn.A*Uy.values).^2))/length(UyEqn.b)/Ry₀

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

# function correct_face_velocity!(Uf, p, )
#     mesh = Uf.mesh
#     (; cells, faces) = mesh
#     nbfaces = total_boundary_faces(mesh)
#     for fID ∈ (nbfaces + 1):length(faces)
#         face = faces[fID]
#         gradp = 0.0
#         Uf.x = nothing
#         ################
#         # CONTINUE 
#         ################
#     end
# end

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

# function correct_boundary_Hvf!(Hvf, ux, uy, ∇pf, UBCs)
#     mesh = ux.mesh
#     for BC ∈ UBCs
#         if typeof(BC) <: Neumann
#             bi = boundary_index(mesh, BC.name)
#             boundary = mesh.boundaries[bi]
#             correct_flux_boundary!(BC, phif, phi, boundary, faces)
#         end
#     end
# end

# function correct_flux_boundary!(
#     BC::Neumann, phif::FaceScalarField{I,F}, phi, boundary, faces) where {I,F}
#     (; facesID, cellsID) = boundary
#     for fID ∈ facesID
#         phif.values[fID] = BC.value 
#     end
# end

function inverse_diagonal!(rD::ScalarField{I,F}, eqn) where {I,F}
    (; mesh, values) = rD
    cells = mesh.cells
    A = eqn.A
    @inbounds for i ∈ eachindex(values)
        D = view(A, i, i)[1]
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
        # values[i] = alpha*values[i] + (1.0 - alpha)*phi0[i]
        values[i] = phi0[i] + alpha*(values[i] - phi0[i])
        phi0[i] = values[i]
    end
end

function correct_velocity!(U, Hv, ∇p, rD)
    # @. U.x = Hv.x - ∇p.x*rD.values
    # @. U.y = Hv.y - ∇p.y*rD.values
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(Ux)
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    # @. ux.values = Hv.x - ∇p.x*rD.values
    # @. uy.values = Hv.y - ∇p.y*rD.values
    u = ux.values; v = uy.values; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(u)
        rDvalues_i = rDvalues[i]
        u[i] = Hvx[i] - dpdx[i]*rDvalues_i
        v[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function neg!(∇p)
    # ∇p.x .*= -1.0
    # ∇p.y .*= -1.0
    dpdx = ∇p.x; dpdy = ∇p.y
    @inbounds for i ∈ eachindex(dpdx)
        dpdx[i] *= -1.0
        dpdy[i] *= -1.0
    end
end

function remove_pressure_source!(ux_eqn, uy_eqn, ∇p, rD)
    # @. ux_eqn.b -= ∇p.x/rD.values
    # @. uy_eqn.b -= ∇p.y/rD.values
    dpdx, dpdy, rD = ∇p.x, ∇p.y, rD.values
    bx, by = ux_eqn.b, uy_eqn.b
    @inbounds for i ∈ eachindex(bx)
        # rDi = rD[i]
        # bx[i] -= dpdx[i]/rDi
        # by[i] -= dpdy[i]/rDi
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
        # rD = 1.0/Ax[cID, cID]
        # x[cID] = (bx[cID] - sumx)*rD
        # y[cID] = (by[cID] - sumy)*rD
        # z[cID] = zero(F)

        # rD_temp = rD.values[cID]/cells[cID].volume # works
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
        # rD = 1.0/Ax[cID, cID]
        # x[cID] = (bx[cID] - sumx)*rD
        # y[cID] = (by[cID] - sumy)*rD
        # z[cID] = zero(F)

        # rD_temp = rD.values[cID]/cells[cID].volume # works
        D = view(Ax, cID, cID)[1] # Good for now (add check to use max of Ax or Ay)
        rD_temp = 1.0/D
        x[cID] = (bx[cID] - sumx)*rD_temp
        y[cID] = (by[cID] - sumy)*rD_temp
        z[cID] = zero(F)
    end
end